import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from modules.gaussian_smoothing import GaussianSmoothing


def ss_warp(x, flo, requires_grad):
    """
    warp an scaled space volume (x) back to im1, according to scale space flow
    x: [B, C, D, H, W]
    flo: [B, 3, 1, H, W] ss flow
    """
    B, _, _, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    zz = torch.zeros_like(xx)
    grid = torch.cat((xx, yy, zz), 1).float()
    grid = grid.unsqueeze(2).to(x.device)

    if requires_grad: grid.requires_grad = True
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0
    vgrid[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :].clone() - 1.0

    vgrid = vgrid.permute(0, 2, 3, 4, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output.squeeze(2)

# 修改了高斯平滑核所接收的通道数 3 → 64，匹配经映射到特征域后的通道维度
class ScaledSpaceFlow(nn.Module):
    def __init__(self, c_in, c_feat, ss_sigma, ss_levels, kernel_size):
        super().__init__()

        self.gaussian_kernels = nn.ModuleList(
            [GaussianSmoothing(64, kernel_size, (2**i) * ss_sigma) for i in range(ss_levels)]
        )

        self.down_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in*2,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),
        )

        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=c_feat,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),
        )

        # self.down_modules = [module for k, module in self.down_layers._modules.items()]
        # self.up_modules = [module for k, module in self.up_layers._modules.items()]

    def generate_ss_volume(self, x):
        out = [x]
        # 第一个尺度，即原始输入特征 x
        for kernel in self.gaussian_kernels:
            out.append(kernel(x))
            # 对输入特征 x 应用每一个高斯卷积核，并将结果添加到输出列表中
        out = torch.stack(out, dim=2)
        # 将所有尺度的结果堆叠在一起（新的第三维度），形成尺度空间特征体积
        return out
    # 假设输入的特征图 x 形状是 (batch_size, c_in, height, width)
    # 尺度空间特征体积生成后，张量形状会是：(batch_size, c_in, ss_levels + 1, height, width)（多1是原始输入）
    # 尺度是通过改变模糊程度而非空间分辨率来实现的

    def forward(self, x):
        # x = x.requires_grad_()
        # down_out = checkpoint.checkpoint_sequential(self.down_modules, len(self.down_modules), x)
        # down_out = down_out.requires_grad_()
        # up_out = checkpoint.checkpoint_sequential(self.up_modules, len(self.up_modules), down_out)

        down_out = self.down_layers(x)
        up_out = self.up_layers(down_out)
        return up_out
