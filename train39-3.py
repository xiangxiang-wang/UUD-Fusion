#  Harvard训练
import os
from modules39 import DFF39
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
# from data_loader_mask import MultiFocusDatasetDUTS
# from data_loader_cvc14 import CVC14
from data_loader_harvard import Harvard
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.nn import init
import numpy as np
import random
import time
from utils import save_images2, RGB2YCbCr, adaptive_coefficient3, save_images3
from Loss import LpLssimLoss


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, args):
    device = args.device
    n_epochs = args.n_epochs
    model_dir = args.saveModel_dir
    batch_size = args.batch_size
    train_num = 2796
    writer1 = SummaryWriter(log_dir="log/loss")
    # ------- 1. define model --------
    # define the model

    net = DFF39(args).to(device)
    net.apply(weights_init_xavier)
    # define the loss

    # criterion = LpLssimLoss().to(args.device)
    # L1 = torch.nn.L1Loss().to(args.device)
    mse = torch.nn.MSELoss().to(args.device)
    # smoothL1 = torch.nn.SmoothL1Loss().to(args.device)

    # mse = nn.MSELoss()

    # ------- 2. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------- 3. training process --------
    # ite_num = 0
    running_loss = 0.0
    # ite_num4val = 0

    # pretrain2_path = r"E:/saveModel_dir_pretrain2/epoch_208_loss_0.084979.pth"
    # pretrained_net = torch.load(pretrain2_path)
    # net.load_state_dict(pretrained_net['net'], strict=False)
    # # model_dict = net.state_dict()
    # # state_dict = {k: v for k, v in pretrained_net.items() if k in model_dict.keys()}
    # # model_dict.update(state_dict)
    # # net.load_state_dict(model_dict)
    # print("加载预训练模型成功")
    # frozen_list = []
    # pretrained_net_item = pretrained_net['net']
    # for name, _ in pretrained_net_item.items():
    #     # name = 'fusion.'+name
    #     frozen_list.append(name)
    # # for p in net.named_parameters():
    # #     p_name = p[0][7:]   # 去掉fusion.initial_result.2.bias 的 fusion.
    # #     if p_name in frozen_list:  # 只冻结在名字列表内的权重
    # #         p[1].requires_grad = False
    # # for p in net.named_parameters():
    # #     if p[0] in frozen_list:  # 只冻结在名字列表内的权重
    # #         p[1].requires_grad = False
    # # # ##打印梯度是否冻结
    # # # for p in net.named_parameters():
    # # #     print(f"{p[0]}'s requires_grad is ]{p[1].requires_grad}")
    # # print("冻结预训练模型成功")

    log_dir = model_dir + "epoch_4_loss_2.325648.pth"  ###若需从断点处继续，需要更改此文件为上一次的epoch

    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     net.load_state_dict(checkpoint['state_dict'], strict=False)
    #     # optimizer.load_state_dict(checkpoint['optimizer'])
    #     # scheduler.load_state_dict(checkpoint['scheduler'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print('加载 epoch {} 成功！'.format(start_epoch))
    # else:
    #     start_epoch = 0
    #     print('无保存模型，将从头开始训练！')

    # log_dir = "E:\saveModel_dir_fusion25\epoch_236_loss_7.626684.pth"

    # pretrain6_path = r"E:/saveModel_dir_pretrain6/epoch_152_loss_2.465181.pth"
    # pretrained_net = torch.load(pretrain6_path)
    # pretrain2_path = r"E:/saveModel_dir_pretrain2/epoch_208_loss_0.084979.pth"
    # pretrained_net = torch.load(pretrain2_path)
    # pretrain8_path = r"E:/saveModel_dir_pretrain8/epoch_152_loss_11.211964.pth"
    # pretrained_net = torch.load(pretrain8_path)
    # pretrain8_path = r"E:/saveModel_dir_pretrain9/epoch_69_loss_27.443536.pth"
    # pretrained_net = torch.load(pretrain8_path)

    # checkpoint_path = r'E:\saveModel_dir_pretrain11\epoch_46_loss_1.704864.pth'
    # checkpoint_path = r'E:\saveModel_dir_pretrain16\epoch_110_loss_79.352591.pth'
    # checkpoint_path = r'E:\saveModel_dir_pretrain19\epoch_88_loss_4.852962.pth'
    checkpoint_path = r'E:\saveModel_dir_pretrain19-3\epoch_74_loss_0.429036.pth'
    pretrained_net = torch.load(checkpoint_path)
    # net.load_state_dict(pretrained_net['state_dict'], strict=False)
    net.load_state_dict(pretrained_net['state_dict'], strict=False)
    # model_dict = net.state_dict()
    # state_dict = {k: v for k, v in pretrained_net.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # net.load_state_dict(model_dict)
    print("加载预训练模型成功")
    frozen_list = []
    # pretrained_net_item = pretrained_net['state_dict']
    pretrained_net_item = pretrained_net['state_dict']
    for name, _ in pretrained_net_item.items():
        # name = 'fusion.'+name
        frozen_list.append(name)
    # for p in net.named_parameters():
    #     p_name = p[0][7:]   # 去掉fusion.initial_result.2.bias 的 fusion.
    #     if p_name in frozen_list:  # 只冻结在名字列表内的权重
    #         p[1].requires_grad = False
    for p in net.named_parameters():
        if p[0] in frozen_list:  # 只冻结在名字列表内的权重
            p[1].requires_grad = False
    ##打印梯度是否冻结
    # for p in net.named_parameters():
    #     print(f"{p[0]}'s requires_grad is {p[1].requires_grad}")
    print("冻结预训练模型成功")

    #     checkpoint = torch.load(log_dir)
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print('加载 epoch {} 成功！'.format(start_epoch))

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    # for epoch in range(0, n_epochs):
    for epoch in range(start_epoch, n_epochs):
        net.train()
        t1 = time.time()
        for i, (image1, image2, _) in enumerate(train_loader):
        # for i, (image1, image2) in enumerate(train_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            # label = label.to(device)
            # Y1, Cb1, Cr1 = RGB2YCbCr(image1)
            # Y2, Cb2, Cr2 = RGB2YCbCr(image2)
            # Y3, _, _ = RGB2YCbCr(label)

            # label = label.to(device)
            # t = DFF.diffusion.sample_timesteps(image1.shape[0]).to(device)
            #
            # predicted_noise = net(image1, image2, t)
            # mask_image1 = label.to(device)  # 保留前景
            # mask_image2 = torch.ones_like(mask_image1)
            # mask_image2 = mask_image2 - mask_image1  # 保留背景
            # x_init, predicted_noise, noise = net(image1, image2, mask_image1, mask_image2)
            # x_init, predicted_noise, noise = net(image1, image2)
            # x_init, predicted_noise, noise = net(Y1, Y2, Y3)
            # x_init, predicted_noise, noise = net(image1, image2, label)
            # x_init, x_final, x_t_pre = net(image1, image2)
            # x_init, x_final, x_t_pre = net(image1, image2, label)
            x_init, x_final1, x_final2 = net(image1, image2)
            # x_init = net.autoencoder_decode(x_in
            # x_init = net.autoencoder_decode(x_init)

            # predicted_noise, noise = net(image1, image2)

            # loss = L1(predicted_noise, noise)
            # loss1 = mse(noise, predicted_noise)
            # loss1 = criterion(predicted_noise, noise)
            # loss2 = smoothL1(predicted_noise, noise)
            # l2 = adaptive_coefficient3(loss1, loss2)
            # loss = loss1 + l2 * loss2
            # loss = loss2
            # loss = mse(x_final, x_init)
            # loss = mse(x_final, x_t_pre)
            # loss = smoothL1(x_final1, image2) + smoothL1(x_final2, image1)
            loss = mse(x_final1, image2) + mse(x_final2, image1)
            # loss = mse(x_final1, image1) + mse(x_final2, image2)
            # loss = smoothL1(x_final1, image2) + smoothL1(x_final2, image1)
            # loss = mse(x_final1, image1) + mse(x_final2, image2)
            # loss = mse(x_final1, image2)
            # loss = mse(x_final, image2)
            # loss = mse(x_final, label)

            # loss = mse(noise, predicted_noise) + mse(image1, x_init) + mse(image2, x_init)
            # loss = mse(noise, predicted_noise) + mse(label, x_init)
            # loss = mse(noise, predicted_noise)
            # ssim = criterion(noise, predicted_noise)
            # l1 = L1(noise, predicted_noise)
            # loss = 0.8 * ssim + 0.2 * l1
            # loss1 = mse(noise, predicted_noise)
            # ssim = criterion(x_init, label)
            # l1 = L1(x_init, label)
            # loss2 = ssim + l1
            # loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            if (i + 1) % 8 == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
                    epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))
        writer1.add_scalar('训练损失', running_loss, epoch)

        # check_point = {'net': net.state_dict(),
        #                'optimizer': optimizer.state_dict(),
        #                'scheduler': scheduler.state_dict(),
        #                'epoch': epoch}
        # checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
        # torch.save(check_point, checkpoint_path)
        if epoch % 2 == 0:
            check_point = {  # 'state_dict': net.state_dict(),
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}
            checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
            torch.save(check_point, checkpoint_path)

        # x1 = net.extract_features1(image1)
        # x2 = net.extract_features2(image2)
        #
        # image2 = image2 * mask_image2
        # x1 = net.autoencoder_encode(image1)
        # x2 = net.autoencoder_encode(image2)
        # x_init = torch.max(x1, x2)
        # x_init = torch.cat([image1, image2], dim=1)
        # x_init = net.imagedown(x_init)
        # x_init = net.conv_out_cat(x_init)
        # res1 = torch.abs(x_init - image1)
        # res2 = torch.abs(x_init - image2)
        # x0 = (image1+image2)/2
        # res1 = x0 - image1
        # res2 = x0 - image2
        # x_res = torch.min(res1, res2)
        # x1 = net.autoencoder_encode(image1)
        # x2 = net.autoencoder_encode(image2)
        # x_init = torch.max(x1, x2)

        # x_init = net.autoencoder_encode(x_res)
        # x1 = net.first_stage_model.encoder(Y1)
        # x2 = net.first_stage_model.encoder(Y2)

        # x1 = net.first_stage_model.encoder(image1)
        # x2 = net.first_stage_model.encoder(image2)
        # # x_init = (image1+image2)/2
        # #
        # x_init = torch.max(x1, x2)
        #
        # x_init = net.first_stage_model.decoder(x_init)

        # x_init = net.first_stage_model(image1, image2)

        # x_res1 = image1 - x_init
        # x_res2 = image2 - x_init
        # x_res1 = self.first_stage_model.encoder(x_res1)
        # x_res2 = self.first_stage_model.encoder(x_res2)
        # x_res = torch.max(x_res1, x_res2)
        # x_res = torch.cat([x_res1,x_res2],dim=1)

        # x_init = net.first_stage_model.decoder(x_init)
        # x_res1 = Y1 - x_init
        # x_res2 = Y2 - x_init
        # x_res = torch.max(x_res1, x_res2)
        # x0 = (Y1+Y2)/2
        # x0 = net.first_stage_model.encoder(x0)
        # x_res = x_init - x0

        # x_res1 = image1 - x_init
        # x_res2 = image2 - x_init
        # x_res1 = net.first_stage_model.encoder(x_res1)
        # x_res2 = net.first_stage_model.encoder(x_res2)
        # x_res = torch.cat([x_res1, x_res2],dim=1)

        # x_init1 = net.autoencoder_encode1(x_init)
        # x_init2 = net.autoencoder_encode2(x_init)
        # x_init = (x_init1 + x_init2)/2
        # x_res1 = x1 - x_init1
        # x_res2 = x2 - x_init2
        # x_res = torch.min(x_res1, x_res2)
        # x1 = net.autoencoder_encode1(image1)
        # x2 = n)et.autoencoder_encode2(image2)
        # x_init = (x1+x2)/2
        # x0 = (image1 + image2)/2
        # x_init = net.autoencoder_encode(x0)
        # x_init = torch.max(x1, x2)
        # sampled_images = net.diffusion.sample2(net.unet, x_res, x1, x2, n=image1.shape[0])
        # sampled_images = net.diffusion.sample2(net.unet, x_init, x1, x2, n=image1.shape[0])
        # sampled_images = net.diffusion.sample2(net.unet, x_init, image1, image2, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur3(net.unet, x_init,image1,image2, n=image1.shape[0])

        # x_noise, sampled_images = net.diffusion.sample_from_blur(net.unet, x_init, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, image1, image2, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur4(net.unet, x_res, n=image1.shape[0])

        # sampled_images = net.diffusion.sample2(net.unet, Y1, Y2, n=Y1.shape[0])
        # sampled_images = net.diffusion.sample2(net.unet, x_init, Y1, Y2, n=Y1.shape[0])
        # sampled_images = net.diffusion.sample2(net.unet, x_res, Y1, Y2, n=Y1.shape[0])

        # sampled_images = net.diffusion.sample2(net.unet, x_res, image1, image2, n=image1.shape[0])  # n=batch_size
        # sampled_images = (x_init + sampled_images)/2
        # sampled_images = x_init

        # sampled_images = net.autoencoder_decode(sampled_images)
        # sampled_images = x_init + sampled_images

        # sampled_images = net.first_stage_model.decoder(sampled_images)

        # sampled_images2 = net.autoencoder_decode2(sampled_images)
        # sampled_images = net.diffusion.sample2(net.unet, x1, x2, n=image1.shape[0])  # n=batch_size
        # sampled_images = net.diffusion.sample2(net.unet, image1, image2, n=image1.shape[0])  # n=batch_size
        # sampled_images = torch.cat([sampled_images, sampled_images], dim=1)
        # x = x_init + sampled_images
        # sampled_images = (sampled_images1 + sampled_images2)/2
        # x = x_init
        # x = (image1 + image2)/2 + sampled_images
        # x = sampled_images
        # x = image1
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # sampled_images = x
        # sampled_images = x_init + sampled_images

        # x1 = net.extract_features1(image1)
        # x2 = net.extract_features2(image2)
        # x = torch.cat([x1, x2], dim=1)
        # x_init = net.initial_result(x)
        # sampled_images = x_init + sampled_images
        # sampled_images = x_init

        # sampled_images = torch.clamp((sampled_images + 1.0) / 2.0, min=0.0, max=1.0)
        # sampled_images = (sampled_images * 255).type(torch.uint8)
        # sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
        # sampled_images = (sampled_images * 255).type(torch.uint8)
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # save_images2(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

        # sampled_images = x[0, :, :, :]
        # out_image = transforms.ToPILImage()(torch.squeeze(sampled_images.data.cpu(), 0))
        # out_image_path = os.path.join("results", args.run_name, f"{epoch}.jpg")
        # out_image.save(out_image_path)

        # torch.save(net.state_dict(), model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss))
        running_loss = 0.0
        scheduler.step()
        t2 = time.time()
        print(t2 - t1)

    print('-------------Congratulations! Training Done!!!-------------')


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    parser.add_argument('--run_name', type=str, default='DDPM_fusion38')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size_h', type=int, default=64)
    parser.add_argument('--image_size_w', type=int, default=64)
    # parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    # parser.add_argument('--dataset_dir', type=str, default="D:\DeepLearing\set\set-m")
    parser.add_argument('--dataset_dir', type=str, default='F:/train_dataset/Harvard')
    parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion39-4/')
    return parser.parse_args()  # 后调用parse_args()方法进行解析，解析成功之后即可使用


if __name__ == '__main__':
    training_setup_seed(1)
    args = parse_args()
    transforms_ = [transforms.Resize((64, 64)),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    train_set = Harvard(dataset_dir=args.dataset_dir, transforms_=transforms_)
    dataset_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train(dataset_dataloader, args)
