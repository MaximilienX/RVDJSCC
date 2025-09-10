import os
import ipdb
import numpy as np

from jsonargparse import ArgumentParser, ActionConfigFile

import torch
#from torch.utils.tensorboard import SummaryWriter


def get_params():
    shell_parse = ArgumentParser()
    shell_parse.add_argument('--trainer', type=str, default='', help='trainer for the job')
    shell_parse.add_argument('--comments', type=str, default='', help='comments for the job')
    shell_parse.add_argument('--device', type=str, help='device to run job')
    shell_parse.add_argument('--eval', action='store_true', help='whether to go into eval mode directly')
    shell_parse.add_argument('--resume', action='store_true', help='resume training from last cp epoch')
    shell_parse.add_argument('--config_file', type=str, help='path to the json file containing params')
    shell_params = shell_parse.parse_args()
    return shell_params


def get_trainer(shell_params):
    parser = ArgumentParser()
    parser.add_argument('--trainer', type=str, default='', help='trainer for the job')
    parser.add_argument('--comments', type=str, default='', help='comments for the job')
    parser.add_argument('--device', type=str, help='device to run job')
    parser.add_argument('--eval', action='store_true', help='whether to go into eval mode directly')
    parser.add_argument('--resume', action='store_true', help='resume training from last cp epoch')
    parser.add_argument('--config_file', action=ActionConfigFile, help='path to the yaml file containing params')

    parser.add_argument('--loss', type=str, default='', help='loss function')
    parser.add_argument('--train_epochs', type=int, help='training epochs')

    match shell_params.trainer:
        case 'recursive_coding':
            from trainer.recursive_coding import RecursiveCoding
            parser = RecursiveCoding.get_parser(parser)
            params = parser.parse_args()
            trainer = RecursiveCoding(params.dataset.dataset, params.loss, params.staged_training, params, params.resume)
        case 'vct_recursive_coding':
            from trainer.vct_recursive_coding import VCTRecursiveCoding
            parser = VCTRecursiveCoding.get_parser(parser)
            params = parser.parse_args()
            trainer = VCTRecursiveCoding(params.dataset.dataset, params.loss, params.staged_training, params, params.resume)
        case 'vct_bandwidth_allocation':
            from trainer.vct_bandwidth_allocation import VCTBandwidthAllocation
            parser = VCTBandwidthAllocation.get_parser(parser)
            params = parser.parse_args()
            trainer = VCTBandwidthAllocation(params.dataset.dataset, params.loss, params, params.resume)
        case 'deepwive':
            from trainer.deepwive import DeepWiVe
            parser = DeepWiVe.get_parser(parser)
            params = parser.parse_args()
            trainer = DeepWiVe(params.dataset.dataset, params.loss, params, params.resume)
        case 'deepjscc_q':
            from trainer.deepjscc_q import DeepJSCC_Q
            parser = DeepJSCC_Q.get_parser(parser)
            params = parser.parse_args()
            trainer = DeepJSCC_Q(params.dataset.dataset, params.loss, params, params.resume)
        case 'deepjscec':
            from trainer.deepjscec import DeepJSCEC
            parser = DeepJSCEC.get_parser(parser)
            params = parser.parse_args()
            trainer = DeepJSCEC(params.dataset.dataset, params.loss, params, params.resume)
        case _:
            raise NotImplementedError

    return trainer, params


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    trainer, params = get_trainer(get_params())
    channel_params = params.channel
    print(params)
    print(str(trainer))

    #if params.eval:
    test_snrs = np.arange(channel_params.test_snr[0], channel_params.test_snr[1]+1)
    #else:
        #test_snrs = [channel_params.eval_snr]

    # 定义日志文件路径
    log_file_path = "try0.2.txt"

    # 如果日志文件已存在，先删除它，以避免重复写入旧的内容
    #if os.path.exists(log_file_path):
    #    os.remove(log_file_path)
    
    while trainer.epoch < params.train_epochs and not params.eval:
        trainer.training()
        train_loss, _, train_aux = trainer(channel_params.train_snr)

        trainer.validate()
        val_loss, terminate, val_aux = trainer(channel_params.eval_snr)

            
        # 记录到文本日志
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Epoch {trainer.epoch}\n")
            log_file.write(f"Train Loss: {train_loss:.4f}\n")
            for data_key, value in train_aux.items():
                log_file.write(f"Train {data_key}: {value:.4f}\n")

            log_file.write(f"Validation Loss: {val_loss:.4f}\n")
            for data_key, value in val_aux.items():
                log_file.write(f"Validation {data_key}: {value:.4f}\n")
            log_file.write("\n")

        if terminate or trainer.epoch >= params.train_epochs:
            print('Training complete'); break

    print('Evaluating...')
    # TODO separate eval loop from main
    for snr in test_snrs:
        trainer.evaluate()
        _, _, eval_aux = trainer([snr, snr+1])

        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"当前测试SNR: {snr}\n")
            for data_key, value in eval_aux.items():
                log_file.write(f"{data_key}: {value:.4f}\n")
            log_file.write("\n")
