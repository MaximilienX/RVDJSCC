import ipdb
import os
import numpy as np
from pytorch_msssim import ms_ssim

from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer

from modules.modem import Modem
from modules.channel import Channel
from modules.scheduler import EarlyStopping
from modules.feature_encoder_AF import FeatureEncoderAF, FeatureDecoderAF, Equalizer
from modules.scale_space_flow import ScaledSpaceFlow, ss_warp
from modules.deepwive.bandwidth_agent import BWAllocator

from utils import calc_loss, calc_msssim, calc_psnr, as_img_array
from utils import get_dataloader, get_optimizer, get_scheduler

import multipath
from contextual import ResBlock

class Opt:
    def __init__(self, M, P, S, K, N_pilot, L, decay, is_clip):
        self.M = M              # Nc
        self.P = P              # 每视频帧packet数（设置为1）
        self.S = S              # Ns
        self.K = K              # Lcp
        self.N_pilot = N_pilot  # Np
        self.L = L              # 多径数
        self.decay = decay      # 衰减因子
        self.is_clip = is_clip  # 是否使用Clipping
# OFDM参数配置类

class DeepWiVe(BaseTrainer):
    def __init__(self, dataset, loss, params, resume=False):
        super().__init__('DeepWiVe', dataset, loss, resume, params.device)

        self.epoch = 0
        self.params = params
        self.save_dir = params.save_dir

        self.key_stage = params.key_stage
        self.interp_stage = params.interp_stage
        self.bw_stage = params.bw_stage

        self._get_config(params)
        # 定义日志文件路径
        self.log_file_path = "try0.2.txt"

        opt = Opt(M=256, P=1, S=12, K=16, N_pilot=2, L=8, decay=4, is_clip=False)
        self.opt = opt
        # OFDM参数配置类的实现
        self.cof_fix = multipath.Channel(self.opt, self.device)
        # 实例化一个Channel类，用来固定生成cof

        self.ofdm = multipath.OFDM(opt, self.device, 'Pilot_bit.pt')

    def _get_config(self, params):
        self.job_name = f'{self.trainer}({self.loss})'

        (self.train_loader,
         self.val_loader,
         self.eval_loader), dataset_aux = self._get_data(params.dataset)
        self.frame_dim = dataset_aux['frame_sizes']

        (self.key_encoder, self.key_decoder,
         self.interp_encoder, self.interp_decoder,
         self.ssf_net) = self._get_encoder(params.encoder, self.frame_dim)

        #self.modem = self._get_modem(params.modem)

        #self.channel = self._get_channel(params.channel)
        ##################### 条件上下文编码相关组件：##########################
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            ResBlock(64, 64, 3),
        ).to(self.device)

        self.context_refine = nn.Sequential(
            ResBlock(64, 64, 3),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        ).to(self.device)

        self.flowDecoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        ).to(self.device)

        self.contextualDecoder = nn.Sequential(
            nn.Conv2d(131, 64, 3, stride=1, padding=1),
            ResBlock(64, 64, 3),
            ResBlock(64, 64, 3),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        ).to(self.device)
        ########################################################################

        self.equalizer = Equalizer(
            c_in=152,
            c_feat=256,
            c_out=120,
            feat_dims=(152,15,20),
            ).to(self.device)

        key_modules = [self.key_encoder, self.key_decoder, self.equalizer]
        interp_modules = key_modules + [self.interp_encoder, self.interp_decoder, self.ssf_net, self.feature_extract,
                                        self.context_refine, self.flowDecoder, self.contextualDecoder]

        #bw_modules = [self.bw_allocator,]
        self.key_optimizer, optimizer_aux = get_optimizer(params.optimizer, key_modules)
        self.interp_optimizer, _ = get_optimizer(params.optimizer, interp_modules)
        #self.bw_optimizer, _ = get_optimizer(params.optimizer, bw_modules)
        self.job_name += '_' + optimizer_aux['str']

        self.key_scheduler, scheduler_aux = get_scheduler(self.key_optimizer, params.scheduler)
        self.interp_scheduler, _ = get_scheduler(self.interp_optimizer, params.scheduler)
        #self.bw_scheduler, _ = get_scheduler(self.bw_optimizer, params.scheduler)
        self.job_name += '_' + scheduler_aux['str']

        self.es = EarlyStopping(mode=params.early_stop.mode,
                                min_delta=params.early_stop.delta,
                                patience=params.early_stop.patience,
                                percentage=False)
        self.job_name += '_' + str(self.es)

        self.scheduler_fn = lambda epochs: epochs % (params.early_stop.patience//2) == 0

        if len(params.comments) != 0: self.job_name += f'_Ref({params.comments})'


        if self.resume: self.load_weights()

    def _get_data(self, params):
        (train_loader, val_loader, eval_loader), dataset_aux = get_dataloader(params.dataset, params)
        self.job_name += '_' + str(train_loader)
        train_loader = data.DataLoader(
            dataset=train_loader,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=2,
        )
        val_loader = data.DataLoader(
            dataset=val_loader,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=2,
        )
        eval_loader = data.DataLoader(
            dataset=eval_loader,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=2,
        )
        return (train_loader, val_loader, eval_loader), dataset_aux

    def _get_encoder(self, params, frame_sizes):
        self.gop_size = params.gop_size
        self.feat_dims = (params.c_out, *[dim // 16 for dim in frame_sizes[1:]])
        self.n_bw_chunks = params.n_bw_chunks
        self.ch_uses_per_gop = np.prod(self.feat_dims) // 2
        self.chunk_size = self.feat_dims[0] // params.n_bw_chunks
        key_encoder = FeatureEncoderAF(
            c_in=params.c_in,
            c_feat=params.c_feat,
            c_out=params.c_out
        ).to(self.device)
        key_decoder = FeatureDecoderAF(
            c_in=params.c_out,
            c_feat=params.c_feat,
            c_out=params.c_in,
            feat_dims=self.feat_dims,
        ).to(self.device)
        # 特征通道数和特征维度都要修改
        interp_encoder = FeatureEncoderAF(
            c_in=131,
            c_feat=params.c_feat,
            c_out=params.c_out
        ).to(self.device)
        interp_decoder = FeatureDecoderAF(
            c_in=params.c_out,
            c_feat=params.c_feat,
            c_out=9,
            feat_dims=self.feat_dims,
        ).to(self.device)
        ssf_net = ScaledSpaceFlow(
            c_in=params.c_in,
            c_feat=params.c_feat,
            ss_sigma=params.ss_sigma,
            ss_levels=params.ss_levels,
            kernel_size=3
        ).to(self.device)

        self.job_name += '_' + str(key_encoder) #+ '_' + str(bw_allocator)
        return key_encoder, key_decoder, interp_encoder, interp_decoder, ssf_net #bw_allocator

    
    def OFDM(self, code, SNR):
        N, C, H, W = code.shape

        # 传输同一个视频帧的所有OFDM包时，信道系数H恒定
        cof, _ = self.cof_fix.sample(N, self.opt.P, self.opt.M, self.opt.L)
        
        # 把特征展平（保留batchsize）
        flattened_code = code.contiguous().view(N, -1)

        # 每个包的大小是Ns*Nc*2=6144
        ofdm_packet_size = self.opt.M * self.opt.S * 2

        # 计算需要分多少OFDM包传输,且即便分配带宽为0也至少传一个全0的OFDM包
        count = (flattened_code.size(1) + ofdm_packet_size - 1) // ofdm_packet_size
        if count ==0:
            count = 1
        
        # 针对最后一个OFDM包补零
        padding_size = count * ofdm_packet_size - flattened_code.size(1)
        padded_code = torch.cat([flattened_code, torch.zeros(N, padding_size, device=self.device)], dim=1)

        # 按OFDM包长度分块
        code_ofdm = padded_code.contiguous().view(N, count, ofdm_packet_size)
        # [N, count, 6144]

        #for 对code_ofdm分批次传输
        for i in range(count):
            current = code_ofdm[:, i, :]
            # [N, 6144]
            tx = current.contiguous().view(N, self.opt.P, self.opt.S, self.opt.M, 2)
            # size = [N, 1, 12, 256, 2]            
            tx_c = torch.view_as_complex(tx.contiguous())
            # size = [N, 1, 12, 256]           
            tx_c = F.normalize(tx_c)
        
            # 封装的OFDM + 多径信道
            out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, self.PAPR_cp = self.ofdm(tx_c, SNR=SNR, cof=cof)

            r1 = torch.view_as_real(self.ofdm.pilot).repeat(N,1,1,1,1).view(N,1,-1)
            # size = [N, 256*2]
            r2 = torch.view_as_real(out_pilot).view(N,1,-1) # Np = 2
            # size = [N, 2*256*2]
            r3 = torch.view_as_real(out_sig).view(N,1,-1) # Ns =12
            # size = [N, 12*256*2]

            if i==0:
                r1_cat = r1
                r2_cat = r2
                r3_cat = r3
            else:
                r1_cat = torch.cat([r1_cat,r1], dim=1)
                r2_cat = torch.cat([r2_cat,r2], dim=1)
                r3_cat = torch.cat([r3_cat,r3], dim=1)
            # concat：原始导频，经历噪声的导频，噪声信号
        
        r1_cat = r1_cat.view(N,-1)
        r2_cat = r2_cat.view(N,-1)
        r3_cat = r3_cat.view(N,-1)

        #remove_r1 = padding_size // 12
        #remove_r2 = padding_size // 6
        #keep_r1 = r1_cat
        #r1_cat = r1_cat[:, :]
        #r2_cat = r2_cat[:, :]
        r3_cat= r3_cat[:, :flattened_code.size(1)]

        # 计算需要补零的数量,准备进解码器
        r1_padding_size = 3300 - r1_cat.size(1) 
        r2_padding_size = 6300 - r2_cat.size(1)
        r3_padding_size = 36000 - r3_cat.size(1)

        # 对每个张量补充零
        if r1_padding_size > 0:
            r1_cat = torch.cat([r1_cat, torch.zeros(N, r1_padding_size, device=self.device)], dim=1)
    
        if r2_padding_size > 0:
            r2_cat = torch.cat([r2_cat, torch.zeros(N, r2_padding_size, device=self.device)], dim=1)
    
        if r3_padding_size > 0:
            r3_cat = torch.cat([r3_cat, torch.zeros(N, r3_padding_size, device=self.device)], dim=1)
               
        rx = torch.cat((r1_cat, r2_cat, r3_cat), 1).contiguous().view(N, -1, H, W)
        ######## [N, 302, 15, 20] ######

        return rx

    def _key_process(self, key_frame, batch_snr, batch_bandwidth):
        #ipdb.set_trace()
        code = self.key_encoder(key_frame, batch_snr) # 形状修改为了 → [12, 240, 15, 20] 72000/12 = 6000 / 256 = 23.4375 /2 = 11.71875   12* 256 *2 = 6144
        #code[:, batch_bandwidth:] = 0
        #code_1 = code[:, :batch_bandwidth]
        #code_1 = code[:, :batch_bandwidth]
        # 可变码率，丢弃一定的通道数
        # [12, 100, 15, 20]

        rx = self.OFDM(code, batch_snr.item())

        eq_rx = self.equalizer(rx, batch_snr)
        
        prediction = torch.sigmoid(self.key_decoder(eq_rx, batch_snr))

        # 形状=[N, 3, 240, 320] 恢复为视频帧
        return prediction, code , eq_rx

    def flow_refine(self, ref, flow):
        return self.flowDecoder(torch.cat((flow, ref), 1)) + flow

    def contextgeneration(self, flow, ref):
        flow_refine = self.flow_refine(ref,flow)
        # 精细化后的3维光流
        ref_feature = self.feature_extract(ref)
        # 通过特征提取网络将参考帧映射到特征域处理
        ssf_vol = self.ssf_net.generate_ss_volume(ref_feature)
        warp = ss_warp(ssf_vol, flow_refine.unsqueeze(2), False if self.stage == 'bw' else True)
        # 经过SSFwarp后的参考帧特征
        context = self.context_refine(warp)
        # 同样对上下文也进行精细化操作
        return context


    def _interp_encode(self, target, ref, batch_bandwidth, batch_snr):
        # ref 【batch, 3, 240, 320】
        # FIXME 对参考帧ref经过 (1)【feature_extract】网络，映射到特征域扩展通道数【batch, C, 240, 320】
        # FIXME 得到特征ref_feature，更丰富的通道信息，代替帧在像素域生成的SSF_vol进行warp操作，但所用flow仍然使用像素域)

        #ssf_vol1 = self.ssf_net.generate_ss_volume(ref[0])
        #ssf_vol2 = self.ssf_net.generate_ss_volume(ref[1])
        #ipdb.set_trace()
        flow1 = self.ssf_net(torch.cat((target, ref[0]), dim=1))
        flow2 = self.ssf_net(torch.cat((target, ref[1]), dim=1))
        # 3维（空间尺度）流，和视频帧的尺度是一样的

        context1 = self.contextgeneration(flow1, ref[0])
        context2 = self.contextgeneration(flow2, ref[1])

        #w1 = ss_warp(ssf_vol1, flow1.unsqueeze(2), False if self.stage == 'bw' else True)
        #w2 = ss_warp(ssf_vol2, flow2.unsqueeze(2), False if self.stage == 'bw' else True)

        #r1 = target - w1
        #r2 = target - w2
        #interp_input = torch.cat((target, w1, w2, r1, r2, flow1, flow2), dim=1)
        interp_input = torch.cat((target,context1,context2), dim=1)

        code = self.interp_encoder(interp_input, batch_snr)
        #code_1 = code[:, :batch_bandwidth]
        #code[:, batch_bandwidth:] = 0
        # 修改成丢弃通道，而不是置0
        return code

    def reconstruction(self, decoder_out, context1, context2):
        prediction = self.contextualDecoder( torch.cat((decoder_out,context1,context2), dim=1) )
        return prediction


    def _interp_decode(self, code, ref, batch_snr):
        decoder_out = self.interp_decoder(code, batch_snr)
        d, flow1, flow2 = torch.chunk(decoder_out, chunks=3, dim=1)

        #f1, f2, a, r = torch.chunk(decoder_out, chunks=4, dim=1)
        #a = F.softmax(a, dim=1)
        #a1, a2, a3 = torch.chunk(a, chunks=3, dim=1)
        #r = torch.sigmoid(r)

        #a1 = a1.repeat_interleave(3, dim=1)
        #a2 = a2.repeat_interleave(3, dim=1)
        #a3 = a3.repeat_interleave(3, dim=1)

        context1 = self.contextgeneration(flow1, ref[0])
        context2 = self.contextgeneration(flow2, ref[1])
        prediction = self.reconstruction(d, context1, context2)

        #ssf_vol1 = self.ssf_net.generate_ss_volume(ref[0])
        #ssf_vol2 = self.ssf_net.generate_ss_volume(ref[1])

        #pred_1 = ss_warp(ssf_vol1, f1.unsqueeze(2), False if self.stage == 'bw' else True)
        #pred_2 = ss_warp(ssf_vol2, f2.unsqueeze(2), False if self.stage == 'bw' else True)
        #prediction = a1 * pred_1 + a2 * pred_2 + a3 * r
        return prediction

    def _get_gop_struct(self, n_frames):
        match self.gop_size:
            case 4:
                interp_struct = [2, 1, 3]
                interp_dist = [2, 1, 1]
                gop_idxs = np.arange(1, n_frames+1, self.gop_size)
            case _:
                raise NotImplementedError
        return interp_struct, interp_dist, gop_idxs
    

    def __call__(self, snr, *args, **kwargs):
        self.check_mode_set()

        terminate = False
        epoch_trackers = {
            'loss_hist': [],
            'dist_loss_hist': [],
            'psnr_hist': [],
            'msssim_hist': [],
        }
        def log_message(log_file_path, epoch, mode, stage, loss_mean):
            # 将训练或验证信息记录到日志文件
            #with open(log_file_path, "a") as log_file:
            #    log_file.write(f"Epoch: {epoch}, 模式: {mode}, 阶段: {stage}, Loss Mean: {loss_mean:.4f}\n")
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Epoch: {epoch}, 模式: {mode}, 阶段: {stage}, Loss Mean: {loss_mean:.4f}\n")

        # 针对A800服务器日志输出频率过高问题做一下进度条修改
        loader_len = len(self.loader)
    #with tqdm(self.loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}', miniters=loader_len) as tepoch:
        import time
        from datetime import timedelta

        time_s = time.time()
        print(self.epoch)
        print("epoch:",self.epoch, "mode:",self.mode, "stage:",self.stage)
        for batch_idx, (frames, vid_fns) in enumerate(self.loader):
            
            #pbar_desc = f'epoch: {self.epoch}, {self.mode} [{self.stage}]'
            #tepoch.set_description(pbar_desc)
            #with tqdm(self.loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            #    for batch_idx, (frames, vid_fns) in enumerate(tepoch):
            #        pbar_desc = f'epoch: {self.epoch}, {self.mode} [{self.stage}]'
            #        tepoch.set_description(pbar_desc)

            epoch_postfix = OrderedDict()
            batch_trackers = {
                'batch_loss': [],
                'batch_dist_loss': [],
                'batch_psnr': [],
                'batch_msssim': [],
            }
            
            n_frames = frames.size(1) // 3
            frames = list(torch.chunk(frames.to(self.device), chunks=n_frames, dim=1))
            frames = [frame.squeeze(1) for frame in frames]
            # frames列表： [25*(batch, 3, 240, 320)]

            match self.stage:
                case 'key':
                    batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                    epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                    batch_bandwidth = int(np.random.randint(self.n_bw_chunks+1) * self.chunk_size)
                    _, _, gop_idxs = self._get_gop_struct(n_frames) #gop_idxs=[ 1,  5,  9, 13, 17, 21, 25]
                    rand_gop = np.random.randint(len(gop_idxs)-1)
                    i_start = gop_idxs[rand_gop]
                    i_end = gop_idxs[rand_gop+1]
                    key_frame = torch.cat(frames[i_start:i_end], dim=0) # 形状=[12（gop为4×batch为3）, 3, 240, 320]

                    
                    predicted_frame , tx_code , rx_code = self._key_process(key_frame, batch_snr, batch_bandwidth)
                    loss, batch_trackers = self._get_loss([predicted_frame], [key_frame], [tx_code], [rx_code], batch_trackers)
                    if self._training: self._update_params(loss)
                case 'interp':
                    batch_bandwidth = int(np.random.randint(self.n_bw_chunks+1) * self.chunk_size)
                    interp_struct, interp_dist, gop_idxs = self._get_gop_struct(n_frames)
                    predictions = []


                    for gop_idx, (i_start, i_end) in enumerate(zip(gop_idxs[:-1], gop_idxs[1:])):
                        batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                        epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                        gop = frames[i_start-1:i_end]
                        gop_predictions = [torch.zeros(1) for _ in range(len(gop))]
                        # 新增对于均衡器损失计算用的潜在编码的列表
                        tx_codes = [torch.zeros(1) for _ in range(len(gop))]
                        rx_codes = [torch.zeros(1) for _ in range(len(gop))]

                        if gop_idx == 0:
                            # NOTE init frame uses full bw
                            init_frame = frames[0]
                            init_prediction , tx_code , rx_code = self._key_process(init_frame, batch_snr, int(self.n_bw_chunks*self.chunk_size))
                            #loss, batch_trackers = self._get_loss([init_prediction], [init_frame], [tx_code], [rx_code], batch_trackers)
                            #if self._training: self._update_params(loss)
                            predictions.append(init_prediction.detach())

                        first_key = predictions[i_start-1]
                        gop_predictions[0] = first_key
                        # 关键帧的潜在变量也要存入对应列表
                        tx_codes[0] = tx_code
                        rx_codes[0] = rx_code

                        last_key = gop[-1]
                        last_key_prediction , tx_code , rx_code = self._key_process(last_key, batch_snr, batch_bandwidth)
                        gop_predictions[-1] = last_key_prediction
                        # 关键帧的潜在变量也要存入对应列表
                        tx_codes[-1] = tx_code
                        rx_codes[-1] = rx_code
                        

                        for (pred_idx, t) in zip(interp_struct, interp_dist):
                            target_frame = gop[pred_idx]
                            ref = (gop_predictions[pred_idx-t], gop_predictions[pred_idx+t])
                            code = self._interp_encode(target_frame, ref, batch_bandwidth, batch_snr)
                            # 已对编码器做了修改：code是按带宽丢弃了部分通道的编码

                            #symbols = self.modem.modulate(code)
                            #rx_symbols, _ = self.channel(symbols, [batch_snr.item()])
                            #demod_symbols = self.modem.demodulate(rx_symbols)

                            # 用OFDM+多径信道代替原有AWGN信道
                            rx = self.OFDM(code, batch_snr.item())

                            rx_code = self.equalizer(rx,batch_snr)

                            prediction = self._interp_decode(rx_code, ref, batch_snr)

                            gop_predictions[pred_idx] = prediction
                            tx_codes[pred_idx] = code
                            rx_codes[pred_idx] = rx_code

                        ##### 之所以要[1:] 原因是第一帧计算图已经参与过损失计算了，梯度已经清零了，所以潜在变量也要去除第一帧
                        loss, batch_trackers = self._get_loss(gop_predictions[1:], gop[1:], tx_codes[1:], rx_codes[1:], batch_trackers)
                        if self._training: self._update_params(loss)
                        predictions.extend([pred.detach() for pred in gop_predictions[1:]])

                case _:
                    raise ValueError

            epoch_trackers, epoch_postfix = self._update_epoch_postfix(batch_trackers,
                                                                        epoch_trackers,
                                                                        epoch_postfix)
            #tepoch.set_postfix(**epoch_postfix)
        time_e = time.time()
        time_oneepoch = time_e - time_s
        formatted_time = str(timedelta(seconds=round(time_oneepoch)))
        print(f"Time for one epoch: {formatted_time}")
        loss_mean, return_aux = self._get_return_aux(epoch_trackers)
        
        # 写入日志
        log_message(self.log_file_path, self.epoch, self.mode, self.stage, loss_mean)
        
        if self._validate: terminate = self._update_es(loss_mean)

        self.reset()
    
        return loss_mean, terminate, return_aux


    def _get_return_aux(self, epoch_trackers):
        return_aux = {}
        loss_mean = np.nanmean(epoch_trackers['loss_hist'])

        #if self.stage == 'bw':
        #    return_aux['dist_loss'] = np.nanmean(epoch_trackers['dist_loss_hist'])

        if not self._training:
            psnr_mean = np.nanmean(epoch_trackers['psnr_hist'])
            msssim_mean = np.nanmean(epoch_trackers['msssim_hist'])

            if self._validate:
                return_aux['psnr_mean'] = psnr_mean
                return_aux['msssim_mean'] = msssim_mean

            elif self._evaluate:
                psnr_std = np.sqrt(np.var(epoch_trackers['psnr_hist']))
                msssim_std = np.sqrt(np.var(epoch_trackers['msssim_hist']))

                return_aux['psnr_mean'] = psnr_mean
                return_aux['psnr_std'] = psnr_std
                return_aux['msssim_mean'] = msssim_mean
                return_aux['msssim_std'] = msssim_std
        return loss_mean, return_aux

    def _update_epoch_postfix(self, batch_trackers, epoch_trackers, epoch_postfix):
        #if self.stage == 'bw':
        #    if len(batch_trackers['batch_loss']) > 0:
        #        epoch_trackers['loss_hist'].append(np.nanmean(batch_trackers['batch_loss']))
        #        epoch_postfix['Q loss'] = '{:.5f}'.format(epoch_trackers['loss_hist'][-1])

        #    epoch_trackers['dist_loss_hist'].append(np.nanmean(batch_trackers['batch_dist_loss']))
        #    epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(epoch_trackers['dist_loss_hist'][-1])
        #else:
        epoch_trackers['loss_hist'].append(np.nanmean(batch_trackers['batch_loss']))
        epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(epoch_trackers['loss_hist'][-1])

        if not self._training:
            epoch_trackers['psnr_hist'].extend(batch_trackers['batch_psnr'])
            batch_psnr_mean = np.nanmean(batch_trackers['batch_psnr'])
            epoch_postfix['psnr'] = '{:.5f}'.format(batch_psnr_mean)

            epoch_trackers['msssim_hist'].extend(batch_trackers['batch_msssim'])
            batch_msssim_mean = np.nanmean(batch_trackers['batch_msssim'])
            epoch_postfix['msssim'] = '{:.5f}'.format(batch_msssim_mean)
        return epoch_trackers, epoch_postfix

    def _get_loss(self, predicted_frames, target_frames, tx_codes, rx_codes, batch_trackers):
        predictions = torch.stack(predicted_frames, dim=1)
        target = torch.stack(target_frames, dim=1)
        tx_code = torch.stack(tx_codes, dim=1)
        rx_code = torch.stack(rx_codes, dim=1)

        match self.stage:
            case 'key':
                loss_frame, _ = calc_loss(predictions, target, self.loss)
                loss_code,  _ = calc_loss(tx_code, rx_code, self.loss)
                loss = loss_frame + 0.2*loss_code
                batch_trackers['batch_loss'].append(loss.item())

                if not self._training:
                    frame_psnr = calc_psnr(predicted_frames, target_frames)
                    batch_trackers['batch_psnr'].extend(frame_psnr)

                    frame_msssim = calc_msssim(predicted_frames, target_frames)
                    batch_trackers['batch_msssim'].extend(frame_msssim)
            case 'interp':
                loss_frame, _ = calc_loss(predictions, target, self.loss)
                loss_code,  _ = calc_loss(tx_code, rx_code, self.loss)
                loss = loss_frame + 0.2*loss_code
                batch_trackers['batch_loss'].append(loss.item())

                if not self._training:
                    frame_psnr = calc_psnr(predicted_frames, target_frames)
                    batch_trackers['batch_psnr'].extend(frame_psnr)

                    frame_msssim = calc_msssim(predicted_frames, target_frames)
                    batch_trackers['batch_msssim'].extend(frame_msssim)

            case _:
                raise ValueError
        return loss, batch_trackers

    def _update_es(self, loss):
        # 日志写入
        def log_message(message):
            with open(self.log_file_path, "a") as log_file:
                log_file.write(message + "\n")
        flag, best_loss, best_epoch, bad_epochs = self.es.step(torch.Tensor([loss]), self.epoch)
        if flag:
            match self.stage:
                case 'key':
                    flag = False
                    self.load_weights()

                    self.key_stage = self.epoch
                    self.stage = 'interp'
                    self.es.reset()
                #case 'interp':
                #    flag = False
                #    self.load_weights()

                #    self.interp_stage = self.epoch
                #    self.stage = 'bw'
                #    self.es.reset()
                case 'interp':
                    print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
                    log_message(f"ES criterion met; loading best weights from epoch {best_epoch}")
                    #新增写入文本日志
        else:
            if bad_epochs == 0:
                self.save_weights()
                print('Saving best weights')
                log_message("Saving best weights.")
                #新增写入日志
            elif self.scheduler_fn(bad_epochs):
                self.lr_scheduler.step()
                lr = self.lr_scheduler.get_last_lr()[0]
                log_message(f"lr updated: {lr:.7f}")
                # 写入学习率
                print('lr updated: {:.7f}'.format(self.lr_scheduler.get_last_lr()[0]))
                es_status = (f"ES status: best: {best_loss.item():.6f}; "
                     f"bad epochs: {bad_epochs}/{self.es.patience}; "
                     f"best epoch: {best_epoch}")
                log_message(es_status)
            print('ES status: best: {:.6f}; bad epochs: {}/{}; best epoch: {}'
                  .format(best_loss.item(), bad_epochs, self.es.patience, best_epoch))
        return flag

    # FIXME 添加新的网络模块时，这里要做修改
    def _set_mode(self):
        match self.mode:
            case 'train':
                self.epoch += 1
                torch.set_grad_enabled(True)
                self.key_encoder.train()
                self.key_encoder.requires_grad_(True)

                self.key_decoder.train()
                self.key_decoder.requires_grad_(True)

                self.equalizer.train()
                self.equalizer.requires_grad_(True)

                self.interp_encoder.train()
                self.interp_encoder.requires_grad_(True)

                self.interp_decoder.train()
                self.interp_decoder.requires_grad_(True)

                self.ssf_net.train()
                self.ssf_net.requires_grad_(True)

                self.feature_extract.train()
                self.feature_extract.requires_grad_(True)
                
                self.context_refine.train()
                self.context_refine.requires_grad_(True)

                self.flowDecoder.train()
                self.flowDecoder.requires_grad_(True)

                self.contextualDecoder.train()
                self.contextualDecoder.requires_grad_(True)

                self.loader = self.train_loader
                self._set_stage()
            case 'val':
                torch.set_grad_enabled(False)
                self.key_encoder.eval()
                self.key_encoder.requires_grad_(False)

                self.key_decoder.eval()
                self.key_decoder.requires_grad_(False)

                self.equalizer.eval()
                self.equalizer.requires_grad_(False)

                self.interp_encoder.eval()
                self.interp_encoder.requires_grad_(False)

                self.interp_decoder.eval()
                self.interp_decoder.requires_grad_(False)

                self.ssf_net.eval()
                self.ssf_net.requires_grad_(False)

                self.feature_extract.eval()
                self.feature_extract.requires_grad_(False)
                
                self.context_refine.eval()
                self.context_refine.requires_grad_(False)

                self.flowDecoder.eval()
                self.flowDecoder.requires_grad_(False)

                self.contextualDecoder.eval()
                self.contextualDecoder.requires_grad_(False)

                self.loader = self.val_loader
                self._set_stage()
            case 'eval':
                torch.set_grad_enabled(False)
                self.key_encoder.eval()
                self.key_encoder.requires_grad_(False)

                self.key_decoder.eval()
                self.key_decoder.requires_grad_(False)

                self.equalizer.eval()
                self.equalizer.requires_grad_(False)

                self.interp_encoder.eval()
                self.interp_encoder.requires_grad_(False)

                self.interp_decoder.eval()
                self.interp_decoder.requires_grad_(False)

                self.ssf_net.eval()
                self.ssf_net.requires_grad_(False)

                self.feature_extract.eval()
                self.feature_extract.requires_grad_(False)
                
                self.context_refine.eval()
                self.context_refine.requires_grad_(False)

                self.flowDecoder.eval()
                self.flowDecoder.requires_grad_(False)

                self.contextualDecoder.eval()
                self.contextualDecoder.requires_grad_(False)

                self.loader = self.eval_loader
                self.stage = 'interp'

    def _set_stage(self):
        if self.epoch <= self.key_stage:
            self.stage = 'key'
            self.optimizer = self.key_optimizer
            self.lr_scheduler = self.key_scheduler

            self.interp_encoder.eval()
            self.interp_encoder.requires_grad_(False)

            self.interp_decoder.eval()
            self.interp_decoder.requires_grad_(False)

            self.ssf_net.eval()
            self.ssf_net.requires_grad_(False)

            self.feature_extract.eval()
            self.feature_extract.requires_grad_(False)
                
            self.context_refine.eval()
            self.context_refine.requires_grad_(False)

            self.flowDecoder.eval()
            self.flowDecoder.requires_grad_(False)

            self.contextualDecoder.eval()
            self.contextualDecoder.requires_grad_(False)
        elif self.epoch <= self.interp_stage:
            self.stage = 'interp'
            self.optimizer = self.interp_optimizer

            self.lr_scheduler = self.interp_scheduler
            self.key_encoder.eval()
            for param in self.key_encoder.parameters():
                param.requires_grad = False

            self.key_decoder.eval()
            for param in self.key_decoder.parameters():
                param.requires_grad = False

            self.equalizer.eval()
            for param in self.equalizer.parameters():
                param.requires_grad = False
            
            if self.epoch == self.key_stage+1: self.es.reset()
            ## 注意如果使用resume训练，且在interp阶段之初，要修改yaml文件中的key epoch数，以清空早停记录器中的损失

    # FIXME 添加新的网络模块时，这里要做修改
    def save_weights(self):
        if not os.path.exists(self.save_dir):
            print('Creating model directory: {}'.format(self.save_dir))
            os.makedirs(self.save_dir)

        torch.save({
            'stage': self.stage,
            'key_encoder': self.key_encoder.state_dict(),
            'key_decoder': self.key_decoder.state_dict(),
            'interp_encoder': self.interp_encoder.state_dict(),
            'interp_decoder': self.interp_decoder.state_dict(),
            'equalizer': self.equalizer.state_dict(),
            'ssf_net': self.ssf_net.state_dict(),
            'feature_extract':self.feature_extract.state_dict(),
            'context_refine':self.context_refine.state_dict(),
            'flowDecoder':self.flowDecoder.state_dict(),
            'contextualDecoder':self.contextualDecoder.state_dict(),
            #'bw_allocator': self.bw_allocator.state_dict(),
            #'modem': self.modem.state_dict(),
            #'channel': self.channel.state_dict(),
            'key_optimizer': self.key_optimizer.state_dict(),
            'interp_optimizer': self.interp_optimizer.state_dict(),
            #'bw_optimizer': self.bw_optimizer.state_dict(),
            'key_scheduler': self.key_scheduler.state_dict(),
            'interp_scheduler': self.interp_scheduler.state_dict(),
            #'bw_scheduler': self.bw_scheduler.state_dict(),
            'es': self.es.state_dict(),
            'epoch': self.epoch
        }, '{}/{}.pth'.format(self.save_dir, self.job_name))

    def load_weights(self):
        cp = torch.load('{}/{}.pth'.format(self.save_dir, self.job_name), map_location='cpu')

        self.stage = cp['stage']

        self.key_encoder.load_state_dict(cp['key_encoder'])
        self.key_decoder.load_state_dict(cp['key_decoder'])
        self.equalizer.load_state_dict(cp['equalizer'])
        self.interp_encoder.load_state_dict(cp['interp_encoder'])
        self.interp_decoder.load_state_dict(cp['interp_decoder'])
        self.ssf_net.load_state_dict(cp['ssf_net'])
        #self.bw_allocator.load_state_dict(cp['bw_allocator'])

        self.feature_extract.load_state_dict(cp['feature_extract'])
        self.context_refine.load_state_dict(cp['context_refine'])
        self.flowDecoder.load_state_dict(cp['flowDecoder'])
        self.contextualDecoder.load_state_dict(cp['contextualDecoder'])

        self.key_optimizer.load_state_dict(cp['key_optimizer'])
        self.interp_optimizer.load_state_dict(cp['interp_optimizer'])
        #self.bw_optimizer.load_state_dict(cp['bw_optimizer'])

        self.key_scheduler.load_state_dict(cp['key_scheduler'])
        self.interp_scheduler.load_state_dict(cp['interp_scheduler'])
        #self.bw_scheduler.load_state_dict(cp['bw_scheduler'])

        self.es.load_state_dict(cp['es'])
        self.epoch = cp['epoch']
        print('Loaded weights from epoch {}'.format(self.epoch))

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
        parser.add_argument('--key_stage', type=int, help='key frame stage training epochs')
        parser.add_argument('--interp_stage', type=int, help='interpolation stage training epochs')
        parser.add_argument('--bw_stage', type=int, help='bandwidth stage training epochs')

        parser.add_argument('--dataset.dataset', type=str, help='dataset: dataset to use')
        parser.add_argument('--dataset.path', type=str, help='dataset: path to dataset')
        parser.add_argument('--dataset.frames_per_clip', type=int, help='dataset: number of frames to extract from each video')
        parser.add_argument('--dataset.train_batch_size', type=int, help='dataset: training batch size')
        parser.add_argument('--dataset.eval_batch_size', type=int, help='dataset: evaluate batch size')

        parser.add_argument('--optimizer.solver', type=str, help='optimizer: optimizer to use')
        parser.add_argument('--optimizer.lr', type=float, help='optimizer: optimizer learning rate')

        parser.add_argument('--optimizer.lookahead', action='store_true', help='optimizer: to use lookahead')
        parser.add_argument('--optimizer.lookahead_alpha', type=float, help='optimizer: lookahead alpha')
        parser.add_argument('--optimizer.lookahead_k', type=int, help='optimizer: lookahead steps (k)')

        parser.add_argument('--scheduler.scheduler', type=str, help='scheduler: scheduler to use')
        parser.add_argument('--scheduler.lr_schedule_factor', type=float, help='scheduler: multi_lr: reduction factor')

        parser.add_argument('--encoder.c_in', type=int, help='encoder: number of input channels')
        parser.add_argument('--encoder.c_feat', type=int, help='encoder: number of feature channels')
        parser.add_argument('--encoder.c_out', type=int, help='encoder: number of output channels')
        parser.add_argument('--encoder.ss_sigma', type=float, help='encoder: standard deviation of the Gaussian kernel for ssflow')
        parser.add_argument('--encoder.ss_levels', type=int, help='encoder: number of levels for ssflow')
        parser.add_argument('--encoder.gop_size', type=int, help='encoder: number frames in a GoP')
        parser.add_argument('--encoder.n_bw_chunks', type=int, help='encoder: number of chunks to allocate in a GoP')
        parser.add_argument('--encoder.policy_batch_size', type=int, help='encoder: policy training batch size')
        parser.add_argument('--encoder.max_memory_size', type=int, help='encoder: policy experience buffer size')

        parser.add_argument('--modem.modem', type=str, help='modem: modem to use')

        parser.add_argument('--channel.model', type=str, help='channel: model to use')
        parser.add_argument('--channel.train_snr', type=list, help='channel: training snr(s)')
        parser.add_argument('--channel.eval_snr', type=list, help='channel: evaluate snr')
        parser.add_argument('--channel.test_snr', type=list, help='channel: test snr(s)')

        parser.add_argument('--early_stop.mode', type=str, help='early_stop: min/max mode')
        parser.add_argument('--early_stop.delta', type=float, help='early_stop: improvement quantity')
        parser.add_argument('--early_stop.patience', type=int, help='early_stop: number of epochs to wait')
        return parser

    def __str__(self):
        return self.job_name

