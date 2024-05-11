#  扩散和图像融合训练22
import os
from random import random

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
import torchvision.transforms as T
import logging
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torchgeometry as tgm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# 0.3-0.7 0.5-0.8
class DiffusionAndFusion39:
    # def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
    # def __init__(self, noise_steps=50, beta_start=0.1, beta_end=1, img_size_h=256, img_size_w=256, device="cuda"):
    def __init__(self, noise_steps=50, beta_start=0.3, beta_end=0.7, img_size_h=256, img_size_w=256, device="cuda"):
    # def __init__(self, noise_steps=50, beta_start=0.8, beta_end=1.0, img_size_h=256, img_size_w=256, device="cuda"):
    # def __init__(self, noise_steps=50, beta_start=0.01, beta_end=0.1, img_size_h=256, img_size_w=256, device="cuda"):
        # def __init__(self, noise_steps=50, beta_start=0.02, beta_end=0.1, img_size_h=256, img_size_w=256, device="cuda"):
        # def __init__(self, noise_steps=50, beta_start=0.4, beta_end=0.9, img_size_h=256, img_size_w=256, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.device = device
        self.channels = 3
        self.kernel_size = 5
        self.kernel_std = 3
        # self.gaussian_kernels = nn.ModuleList(self.get_kernels()).to(device)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # 连乘 size不变

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  # 线性时间步长，返回值为一维

    def noise_images(self, x_start, x_end, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]  # 扩充维度
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # Ɛ = torch.randn_like(x)  # 噪声z
        img_noise = sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * x_end
        return img_noise

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]  # 扩充维度
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return (xt - sqrt_alpha_hat * x1_bar) / sqrt_one_minus_alpha_hat

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # def blur(self, dims, std):
    #     return tgm.image.get_gaussian_kernel2d(dims, std)
    #
    # def get_conv(self, dims, std, mode='circular'):
    #     kernel = self.blur(dims, std)
    #     conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims,
    #                      padding=int((dims[0] - 1) / 2), padding_mode=mode,
    #                      bias=False, groups=self.channels)
    #     with torch.no_grad():
    #         kernel = torch.unsqueeze(kernel, 0)
    #         kernel = torch.unsqueeze(kernel, 0)
    #         kernel = kernel.repeat(self.channels, 1, 1, 1)
    #         conv.weight = nn.Parameter(kernel)
    #
    #     return conv
    #
    # def get_kernels(self):
    #     kernels = []
    #     for i in range(self.noise_steps):
    #         kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std)))
    #     return kernels
    #
    # def q_sample(self, x_start, t):
    #     max_iters = torch.max(t)
    #     all_blurs = []
    #     x = x_start
    #     for i in range(max_iters + 1):
    #         with torch.no_grad():
    #             x = self.gaussian_kernels[i](x)
    #             all_blurs.append(x)
    #
    #     all_blurs = torch.stack(all_blurs)
    #
    #     choose_blur = []
    #     for step in range(t.shape[0]):
    #         choose_blur.append(all_blurs[t[step], step])
    #
    #     choose_blur = torch.stack(choose_blur)
    #
    #     return choose_blur  # b,c,h,w
    #
    # def sample_from_blur(self, model, img, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps
    #     for i in range(t):  # 0~49
    #         with torch.no_grad():
    #             img = self.gaussian_kernels[i](img)
    #     xt = img
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):  # 49~0
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #         step = (torch.ones(n) * i).long().to(self.device)
    #         with torch.no_grad():
    #             x = model(img, step)
    #             # if i != 0:
    #             #     for j in range(i):  # 0~48
    #             #         with torch.no_grad():
    #             #             x = self.gaussian_kernels[j](x)
    #             # with torch.no_grad():
    #             img = x
    #     model.train()
    #     return xt, img
    #
    # def sample_from_blur2(self, model, img, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps
    #     for i in range(t):
    #         with torch.no_grad():
    #             img = self.gaussian_kernels[i](img)
    #     xt = img
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #         step = (torch.ones(n) * i).long().to(self.device)
    #         x = model(img, step)
    #         if i != 0:
    #             for j in range(i):
    #                 with torch.no_grad():
    #                     x = self.gaussian_kernels[j](x)
    #                     if j == i - 1:
    #                         x_time = self.gaussian_kernels[i](x)
    #         img = x + img - x_time
    #     model.train()
    #     return xt, img
    #
    # def sample_from_blur3(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps
    #     for i in range(t):
    #         with torch.no_grad():
    #             img = self.gaussian_kernels[i](img)
    #     xt = img
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #         step = (torch.ones(n) * i).long().to(self.device)
    #         x = model(img, img1, img2, step)
    #         if i != 0:
    #             for j in range(i):
    #                 with torch.no_grad():
    #                     x = self.gaussian_kernels[j](x)
    #         img = x
    #     model.train()
    #     return xt, img
    #
    # def sample_from_blur4(self, model, img, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps
    #     for i in range(t):
    #         with torch.no_grad():
    #             img = self.gaussian_kernels[i](img)
    #     xt = img
    #     noise = torch.randn_like(img)
    #     img = img + noise
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #         step = (torch.ones(n) * i).long().to(self.device)
    #         x = model(img, step)
    #         if i != 0:
    #             for j in range(i):
    #                 with torch.no_grad():
    #                     x = self.gaussian_kernels[j](x)
    #                     if j == i - 1:
    #                         x_time = self.gaussian_kernels[i](x)
    #         img = x + img - x_time
    #         # img = x
    #     model.train()
    #     return xt, img
    #
    # def sample_from_blur5(self, model, img, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps
    #     for i in range(t):  # 0~49
    #         with torch.no_grad():
    #             img = self.gaussian_kernels[i](img)
    #
    #     noise = torch.randn_like(img)
    #     xt = img + noise
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):  # 49~0
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #         step = (torch.ones(n) * i).long().to(self.device)
    #         with torch.no_grad():
    #             x = model(img, step)
    #             # if i != 0:
    #             #     for j in range(i):  # 0~48
    #             #         with torch.no_grad():
    #             #             x = self.gaussian_kernels[j](x)
    #             # with torch.no_grad():
    #             img = x
    #     model.train()
    #     return xt, img
    #
    # def sample_from_blur6(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps - 1
    #     # t = torch.randint(low=1, high=self.noise_steps, size=(1,))
    #     # for i in range(t):
    #     #     with torch.no_grad():
    #     #         img = self.gaussian_kernels[i](img)
    #     # img1 = self.gaussian_kernels[i](img1)
    #     # img2 = self.gaussian_kernels[i](img2)
    #     # img = (img+img1+img2)/3
    #     # img = img2 - img1
    #     # xt = img
    #     # noise = torch.randn_like(img)
    #     # img = img + noise
    #     # for i in tqdm(reversed(range(1,self.noise_steps+1)), position=0):
    #     while (t):
    #         with torch.no_grad():
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * t).long().to(self.device)
    #             x1_bar = model(img, step)
    #             x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
    #             xt_bar = x1_bar
    #             if t != 0:
    #                 xt_bar = self.noise_images(x_start=xt_bar, x_end=x2_bar, t=step)
    #             xt_sub1_bar = x1_bar
    #             if t - 1 != 0:
    #                 step2 = step - 2
    #                 xt_sub1_bar = self.noise_images(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)
    #
    #             x = img - xt_bar + xt_sub1_bar
    #             img = x
    #             t = t - 1
    #     model.train()
    #     return x1_bar, img
    #
    # def sample_from_blur7(self, model, img,img1,img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps-1
    #     # t = torch.randint(low=1, high=self.noise_steps, size=(1,))
    #     # for i in range(t):
    #     #     with torch.no_grad():
    #     #         img = self.gaussian_kernels[i](img)
    #             # img1 = self.gaussian_kernels[i](img1)
    #             # img2 = self.gaussian_kernels[i](img2)
    #     # img = (img+img1+img2)/3
    #     # img = img2 - img1
    #     # xt = img
    #     # noise = torch.randn_like(img)
    #     # img = img + noise
    #     # for i in tqdm(reversed(range(1,self.noise_steps+1)), position=0):
    #     while(t):
    #         with torch.no_grad():
    #         # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * t).long().to(self.device)
    #             x1_bar = model(img, step)
    #             # x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
    #             xt_bar = x1_bar
    #             if t != 0:
    #                 xt_bar = self.noise_images(x_start=xt_bar, x_end=img1,t=step)
    #             x1_bar = model(xt_bar, step)
    #             xt_sub1_bar = xt_bar
    #
    #             if t-1 !=0:
    #                 step2 = step-2
    #
    #                 xt_sub1_bar = self.noise_images(x_start=xt_sub1_bar,x_end=img2,t=step2)
    #             #
    #             # x = img - xt_bar + xt_sub1_bar
    #             img = xt_sub1_bar
    #             t=t-1
    #     model.train()
    #     return x1_bar, img
    #
    # # def sample_from_blur7(self, model, img, img1, img2, n):
    # #     logging.info(f"Sampling {n} new images....")
    # #     model.eval()
    # #     t = self.noise_steps - 1
    # #     # t = torch.randint(low=1, high=self.noise_steps, size=(1,))
    # #     # for i in range(t):
    # #     #     with torch.no_grad():
    # #     #         img = self.gaussian_kernels[i](img)
    # #     # img1 = self.gaussian_kernels[i](img1)
    # #     # img2 = self.gaussian_kernels[i](img2)
    # #     # img = (img+img1+img2)/3
    # #     # img = img2 - img1
    # #     # xt = img
    # #     # noise = torch.randn_like(img)
    # #     # img = img + noise
    # #     # for i in tqdm(reversed(range(1,self.noise_steps+1)), position=0):
    # #     while (t):
    # #         with torch.no_grad():
    # #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    # #             step = (torch.ones(n) * t).long().to(self.device)
    # #             x1_bar = model(img, step)
    # #             # x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
    # #             xt_bar = x1_bar
    # #             if t - 1 != 0:
    # #                 xt_bar = self.noise_images(x_start=xt_bar, x_end=img1, t=step - 1)
    # #                 # x1_bar = model(xt_bar, step)
    # #                 xt_bar = self.noise_images(x_start=xt_bar, x_end=img2, t=step - 1)
    # #
    # #             img = xt_bar
    # #             t = t - 1
    # #     model.train()
    # #     return x1_bar, img
    #
    # def sample_from_blur8(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps - 1
    #     # t = torch.randint(low=1, high=self.noise_steps, size=(1,))
    #     # for i in range(t):
    #     #     with torch.no_grad():
    #     #         img = self.gaussian_kernels[i](img)
    #     # img1 = self.gaussian_kernels[i](img1)
    #     # img2 = self.gaussian_kernels[i](img2)
    #     # img = (img+img1+img2)/3
    #     # img = img2 - img1
    #     # xt = img
    #     # noise = torch.randn_like(img)
    #     # img = img + noise
    #     # for i in tqdm(reversed(range(1,self.noise_steps+1)), position=0):
    #     while (t):
    #         with torch.no_grad():
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * t).long().to(self.device)
    #             x1_bar = model(img, step)
    #             # x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
    #             xt_bar = x1_bar
    #             if t != 0:
    #                 xt_bar = self.noise_images(x_start=xt_bar, x_end=img1, t=step - 1)
    #             x1_bar = model(xt_bar, step)
    #             xt_sub1_bar = x1_bar
    #
    #             if t != 0:
    #                 step2 = step - 1
    #
    #                 xt_sub1_bar = self.noise_images(x_start=xt_sub1_bar, x_end=img2, t=step2)
    #             #
    #             # x = img - xt_bar + xt_sub1_bar
    #             img = xt_sub1_bar
    #             t = t - 1
    #     model.train()
    #     return x1_bar, img

    def sample_from_blur9(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        t = self.noise_steps - 1

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
                step = (torch.ones(n) * i).long().to(self.device)
                x0_bar = model(img, step)
                x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
                x0_bar = model(x_t_bar1, step)
                # x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
                if i != 0:
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step - 1)
                    #     x_t_bar1 = self.noise_images(x_start=x_t_bar1, x_end=img2, t=step-1)
                    #     # x_t_prv = torch.max(x_t_bar1, x_t_bar2)
                    img = x_t_bar1
                else:
                    img = x0_bar
                #     img1= x_t_bar1

        model.train()
        return x0_bar, img

    # def sample_from_blur10(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps - 1
    #
    #     x_noise1, x_noise2, x0 = [], [], []
    #     x0.append(img)
    #     for i in tqdm(reversed(range(self.noise_steps)), position=0):
    #         with torch.no_grad():
    #
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * i).long().to(self.device)
    #
    #             x0_bar = model(img, step)
    #             x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
    #             x_noise1.append(x_t_bar1)
    #             x0_bar = model(x_t_bar1, step)
    #             # x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
    #             if i !=0:
    #                 x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img2, t=step-1)
    #                 x_noise2.append(x_t_bar1)
    #                 img = x_t_bar1
    #                 # print(i)
    #             else:
    #                 img = x0_bar
    #             #     img1= x_t_bar1
    #             #     print("第0次")
    #             # 必须克隆tensor否则共享内存，to CPU 时 前一个变量移走，后一个变量就为空了
    #             x0_ = img.clone()
    #             x0.append(x0_)
    #
    #
    #     model.train()
    #     return x0_bar, img, x_noise1,x_noise2,x0
    #
    # def sample_from_blur11(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps - 1
    #
    #
    #     for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
    #         with torch.no_grad():
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * i).long().to(self.device)
    #             x0_bar_1 = model(img, step)
    #             x_t_bar1 = self.noise_images(x_start=x0_bar_1, x_end=img1, t=step)
    #             x0_bar_2 = model(x_t_bar1, step)
    #             # x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
    #             # if i !=0:
    #             x_t_bar2 = self.noise_images(x_start=x0_bar_2, x_end=img2, t=step-1)
    #             #     x_t_bar1 = self.noise_images(x_start=x_t_bar1, x_end=img2, t=step-1)
    #             #     # x_t_prv = torch.max(x_t_bar1, x_t_bar2)
    #             img = x_t_bar2
    #             # else:
    #             #     img = x0_bar_1
    #             #     img1= x_t_bar1
    #
    #
    #     model.train()
    #     return x0_bar_2, img
    #
    # def sample_from_blur12(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     t = self.noise_steps - 1
    #
    #
    #     for i in tqdm(reversed(range(1,self.noise_steps)), position=0):
    #         with torch.no_grad():
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * i).long().to(self.device)
    #             x0_bar_1 = model(img, step)
    #             x_t_bar1 = self.noise_images(x_start=x0_bar_1, x_end=img2, t=step)
    #             x0_bar_2 = model(x_t_bar1, step)
    #             x_t_bar2 = self.noise_images(x_start=x0_bar_2, x_end=img1, t=step)
    #             img = model(x_t_bar2, step-1)
    #             # x0_bar_2 = model(x_t_bar1, step)
    #             # x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
    #             # x0_bar_2 = model(x_t_bar1, step)
    #             # if i !=0:
    #             #     x0_bar_2 = model(x_t_bar1, step-1)
    #             #     x_t_bar2 = self.noise_images(x_start=x0_bar_2, x_end=img1, t=step-1)
    #             # #     x_t_bar1 = self.noise_images(x_start=x_t_bar1, x_end=img2, t=step-1)
    #             # #     # x_t_prv = torch.max(x_t_bar1, x_t_bar2)
    #             #     img = x_t_bar2
    #             # else:
    #             #     img = x0_bar_1
    #             # #     img1= x_t_bar1
    #
    #
    #     model.train()
    #     return x0_bar_2, img
    #
    # def sample_from_blur13(self, model, img, img1, img2, n):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     # t = self.noise_steps - 1
    #     img_init = img
    #     for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
    #         with torch.no_grad():
    #             # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
    #             step = (torch.ones(n) * i).long().to(self.device)
    #             x0_bar = model(img, step)
    #             x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
    #             x0_bar2 = model(x_t_bar1, step)
    #             x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img1, t=step)
    #             x0_bar3 = model(x_t_bar2, step)
    #             x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img1, t=step-1)
    #             # img = model(x_t_bar3, step-1)
    #             # x_t_bar4 = self.noise_images(x_start=x0_bar3, x_end=img2, t=step - 1)
    #             # img = img + x_t_bar3 - x_t_bar2 + x_t_bar4 - x_t_bar2
    #             # img = img + x_t_bar3 - img_init + x_t_bar4 - img_init
    #             img = img + x_t_bar3 - x_t_bar2
    #     model.train()
    #     return x0_bar, img

    def sample_from_blur14(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step)
                    x0_bar3 = model(x_t_bar2, step)
                    x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                    img = img + x_t_bar3 - x_t_bar2

                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    img = model(img, step) + img_init
                    # img = model(img, step) + img_init + img2
                    # img = model(img, step)

        model.train()
        return x0_bar, img


    def sample_from_blur14_2(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img
        res = img1-img2
        # res = (img1 - img2)/3

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step) + res
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step) + res
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step-1)
                    # x0_bar3 = model(x_t_bar2, step) + res
                    # x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                    # img = img + x_t_bar3 - x_t_bar2
                    img = x_t_bar2
                    # img = x0_bar

                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    # img = model(img, step) + (img1-2 *img2)
                    # img = model(img, step) + img_init + img2
                    img = model(img, step)

        model.train()
        return x0_bar, img


    def sample_from_blur14_3(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img
        res = img1-img2
        # res = (img1 - img2)/3

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step-1)
                    # x0_bar3 = model(x_t_bar2, step) + res
                    # x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                    # img = img + x_t_bar3 - x_t_bar2
                    img = x_t_bar2-x_t_bar1 +img +res
                    # img = x0_bar

                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    # img = model(img, step) + (img1-2 *img2)
                    # img = model(img, step) + img_init + img2
                    img = model(img, step) + img_init

        model.train()
        return x0_bar, img



    def sample_from_blur15(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img
        img = img1

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:

                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    img = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
                # x0_bar2 = model(x_t_bar1, step)
                # x_t_bar3 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step - 1)
                # # img = img + x_t_bar3 - x_t_bar1
                # img = x_t_bar3
                else:
                    img = model(img, step)

        model.train()
        return x0_bar, img

    def sample_from_blur16(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step)
                    x0_bar3 = model(x_t_bar2, step)
                    x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                    img = img + x_t_bar3 - x_t_bar2

                else:
                    step = (torch.ones(n) * i).long().to(self.device)
                    img = model(img, step)

                    # img = model(img, step) + img1

        model.train()
        return x0_bar, img

    def sample_from_blur17(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img
        x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_ = [], [], [], [], []

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x0_bar_.append(x0_bar)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x0_bar2_.append(x0_bar2)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step)
                    x0_bar3 = model(x_t_bar2, step)
                    x0_bar3_.append(x0_bar3)
                    x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                    x_t_bar3_.append(x_t_bar3)
                    img = img + x_t_bar3 - x_t_bar2

                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    img = model(img, step)
                img_temp = img.clone()
                img_.append(img_temp)

        model.train()
        return x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_, img
    # def sample_from_blur10(self, model, img, img1, img2, n):
    def sample_from_blur18(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img
        x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_ = [], [], [], [], []

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x0_bar_.append(x0_bar)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x0_bar2_.append(x0_bar2)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step)
                    x0_bar3 = model(x_t_bar2, step)
                    x0_bar3_.append(x0_bar3)
                    x_t_bar3 = self.noise_images(x_start=x0_bar2, x_end=img_init, t=step - 1)
                    x_t_bar3_.append(x_t_bar3)
                    img = img

                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    img = model(img, step)
                img_temp = img.clone()
                img_.append(img_temp)

        model.train()
        return x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_, img

    def sample_from_blur19(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                    x0_bar2 = model(x_t_bar1, step)
                    x_t_bar2 = self.noise_images(x_start=x0_bar2, x_end=img2, t=step)
                    x0_bar3 = img + x_t_bar2 - x_t_bar2


                else:
                    step = (torch.ones(n) * i).long().to(self.device)

                    img = model(img, step) + img_init

        model.train()
        return x0_bar, img

    def sample_from_blur20(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # t = self.noise_steps - 1

        img_init = img

        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():
                if i != 0:
                    step = (torch.ones(n) * i).long().to(self.device)
                    x0_bar = model(img, step)
                    if i%2 ==0:
                        x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                        x0_bar2 = model(x_t_bar1, step)
                        x_t_bar3 = self.noise_images(x_start=x0_bar2, x_end=img_init, t=step - 1)
                        img = img + x_t_bar3 - x_t_bar1
                        # img = x0_bar2
                    else:
                        x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
                        x0_bar3 = model(x_t_bar2, step)
                        x_t_bar3 = self.noise_images(x_start=x0_bar3, x_end=img_init, t=step - 1)
                        img = img + x_t_bar3 - x_t_bar2
                        # img = x0_bar3

                else:
                    step = (torch.ones(n) * i).long().to(self.device)
                    # img = model(img, step)
                    img = model(img, step)

        model.train()
        return x0_bar, img

    def sample_from_blur40(self, model, img, img1, img2, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        t = self.noise_steps - 1

        x_noise1, x_noise2, x0 = [], [], []
        x0.append(img)
        for i in tqdm(reversed(range(self.noise_steps)), position=0):
            with torch.no_grad():

                # t = (torch.ones(n) * i).long().to(self.device)  # 批大小的某一时间点，大小为[1,b]
                step = (torch.ones(n) * i).long().to(self.device)

                x0_bar = model(img, step)
                x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img1, t=step)
                # x_noise1.append(x_t_bar1)
                x0_bar = model(x_t_bar1, step)
                # x_t_bar2 = self.noise_images(x_start=x0_bar, x_end=img2, t=step)
                if i !=0:
                    x_t_bar1 = self.noise_images(x_start=x0_bar, x_end=img2, t=step-1)
                    # x_noise2.append(x_t_bar1)
                    img = x_t_bar1
                    # print(i)
                else:
                    img = x0_bar
                #     img1= x_t_bar1
                #     print("第0次")
                # 必须克隆tensor否则共享内存，to CPU 时 前一个变量移走，后一个变量就为空了
                x0_ = img.clone()
                x0.append(x0_)


        model.train()
        return x0_bar, img, x_noise1,x_noise2,x0