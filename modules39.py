# 扩散和图像融合训练22,最朴素
#  扩散和融合的UNet22
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm_fusion39 import DiffusionAndFusion39
from autoencoder19 import Autoencoder19

from utils import save_images2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            # nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
            # return F.relu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels),
#         )
#
#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,
#                 out_channels
#             ),
#         )
#
#     def forward(self, x, t):
#         x = self.maxpool_conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # x = self.conv(x)
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # 防止出现上下采样后维度尺寸不匹配的情况
        _, _, h, w = x.shape
        _, _, skip_h, skip_w = skip_x.shape
        output_padding = padding_img(h, w, skip_h, skip_w)
        x = F.pad(x, output_padding, mode="replicate")
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        # self.ca = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        EPSILON = 1e-10
        # ca1 = self.ca(x1)
        # ca2 = self.ca(x2)
        # mask1 = torch.exp(ca1) / (torch.exp(ca2) + torch.exp(ca1) + EPSILON)
        # mask2 = torch.exp(ca2) / (torch.exp(ca1) + torch.exp(ca2) + EPSILON)
        mask1 = torch.exp(x1) / (torch.exp(x2) + torch.exp(x1) + EPSILON)
        mask2 = torch.exp(x2) / (torch.exp(x1) + torch.exp(x2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a


class UNet_fusion39(nn.Module):
    def __init__(self, c_in=3, c_out=3, img_size_h=64, img_size_w=64, time_dim=256, device="cuda"):
        # def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # self.ca = ChannelAttentionBlock()

        self.inc = DoubleConv(c_in, 32)

        self.down1 = Down(32, 64)

        self.down2 = Down(64, 128)

        self.down3 = Down(128, 128)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)

        self.up2 = Up(128, 32)

        self.up3 = Up(64, 32)

        self.outc = nn.Conv2d(32, c_in, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)

        # _, _, h1, w1 = x1.shape
        # output_padding = padding_img(h, w)
        # x1 = F.pad(x1, output_padding, mode="replicate")

        x2 = self.down1(x1, t)

        # _, _, h2, w2 = x2.shape
        # output_padding = padding_img(h, w)
        # x2 = F.pad(x2, output_padding, mode="replicate")

        x3 = self.down2(x2, t)

        # _, _, h3, w3 = x3.shape
        # output_padding = padding_img(h, w)
        # x3 = F.pad(x3, output_padding, mode="replicate")

        x4 = self.down3(x3, t)

        # _, _, h4, w4 = x4.shape
        # output_padding = padding_img(h, w)
        # x4 = F.pad(x4, output_padding, mode="replicate")

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)

        x = self.up2(x, x2, t)

        x = self.up3(x, x1, t)

        output = self.outc(x)

        return output


class DFF39(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.diffusion = DiffusionAndFusion39(img_size_h=args.image_size_h, img_size_w=args.image_size_w,
                                              device=self.device)
        self.unet = UNet_fusion39(img_size_h=args.image_size_h, img_size_w=args.image_size_w)
        self.ch_att = ChannelAttentionBlock()
        self.first_stage_model = Autoencoder19(in_channels=3, out_channels=3)

    def forward(self, img1, img2):
        t = self.diffusion.sample_timesteps(img1.shape[0]).to(self.device)  # [b]

        x_init = self.first_stage_model(img1, img2)
        # x_init = img

        # x_res1 = img1 - x_init
        # x_res2 = img2 - x_init
        # x_res = torch.max(x_res1, x_res2)
        # x_res = label - x_init

        # x_t_noise = self.diffusion.q_sample(x_init, t)
        # x_t_pre = self.diffusion.q_sample(x_init, t_pre)
        # x_t_noise = self.diffusion.q_sample(x_res, t)
        # x_t_noise1 = self.diffusion.q_sample(img1, t)
        # x_t_noise2 = self.diffusion.q_sample(img2, t)
        x_t_noise1 = self.diffusion.noise_images(img1, img2, t)
        x_t_noise2 = self.diffusion.noise_images(img2, img1, t)
        # x_t_noise1 = self.diffusion.noise_images(img1, x_init, t)
        # x_t_noise2 = self.diffusion.noise_images(img2, x_init, t)

        # x_final1 = self.unet(x_t_noise1, t)
        # x_final2 = self.unet(x_t_noise2, t)
        x_final1 = self.unet(x_t_noise1, t)
        x_final2 = self.unet(x_t_noise2, t)
        # x_final1 = self.unet(img1, t)
        # x_final2 = self.unet(img2, t)
        # x_final2 = x_final1

        # x_final = self.unet(x_t_noise, t)
        # x_final = self.unet(x_t_noise, img1, img2, t)

        # x_t_noise, noise = self.diffusion.noise_images(x_init, t)
        # predicted_noise = self.unet(x_t_noise, x1, x2, t)
        # predicted_noise = self.unet(x_t_noise, img1, img2, t)
        # predicted_noise = self.unet(x_t_noise, mask1, img2, t)

        return x_init, x_final1, x_final2
        # return x_init, x_final, x_t_pre

        # return predicted_noise, noise


def padding_img(h, w, skip_h, skip_w):
    if (h == skip_h) and (w == skip_w):  # 左右上下
        output_padding = (0, 0, 0, 0)
    elif (h != skip_h) and (w == skip_w):  # 左右上下
        output_padding = (0, 0, 0, 1)
    elif (h == skip_h) and (w != skip_w):  # 左右上下
        output_padding = (0, 1, 0, 0)
    else:  # (h % 2 == 1) and (w % 2 == 1)
        output_padding = (0, 1, 0, 1)
    return output_padding
