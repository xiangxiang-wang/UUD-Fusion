#  扩散和图像融合预训练12
import os
from model_pretrain19 import Pretrain19
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
# from data_loader_mask import MultiFocusDatasetDUTS
# from data_loader import MultiFocusDatasetDUTS
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
from utils import compute_loss, adaptive_coefficient, adaptive_coefficient2, adaptive_coefficient3
from Loss import LpLssimLoss
# from utils import YCrCb2RGB, RGB2YCrCb
from utils import RGB2YCbCr

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
# def weights_init_xavier(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         init.xavier_uniform_(m.weight.data)

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
    train_num = 368 * 2
    writer1 = SummaryWriter(log_dir="log/loss")
    # ------- 1. define model --------
    # define the model

    net = Pretrain19().to(device)
    net.apply(weights_init_xavier)
    # define the loss

    criterion = LpLssimLoss().to(args.device)
    L1 = torch.nn.L1Loss().to(args.device)
    mse = torch.nn.MSELoss().to(args.device)
    smoothL1 = torch.nn.SmoothL1Loss().to(args.device)

    # mse = nn.MSELoss()

    # ------- 2. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------- 3. training process --------
    # ite_num = 0
    running_loss = 0.0
    # ite_num4val = 0

    #     premask_path = r"D:/DeepLearing/pythonProject/code/permask/best.pth"
    #     pretrained_net = torch.load(premask_path)
    #     model.load_state_dict(pretrained_net['net'], strict=False)
    #     # model_dict = net.state_dict()
    #     # state_dict = {k: v for k, v in pretrained_net.items() if k in model_dict.keys()}
    #     # model_dict.update(state_dict)
    #     # net.load_state_dict(model_dict)
    #     print("加载预训练模型成功")
    #     frozen_list = []
    #     for name, _ in pretrained_net.items():
    #         frozen_list.append(name)
    #     for p in model.named_parameters():
    #         if p[0] in frozen_list:  # 只冻结在名字列表内的权重
    #             p[1].requires_grad = False
    # ##打印梯度是否冻结
    #     # for p in net.named_parameters():  ++

    #     #     print(f"{p[0]}'s requires_grad is {p[1].requires_grad}")
    #     print("冻结预训练模型成功")

    log_dir = model_dir + "epoch_62_loss_0.497066.pth"  ###若需从断点处继续，需要更改此文件为上一次的epoch
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
        for i, (image1, image2, label) in enumerate(train_loader):
        # for i, (image1, image2) in enumerate(train_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            # image1 = RGB2YCrCb(image1, device)
            # image2 = RGB2YCrCb(image2, device)
            # Y1, Cb1, Cr1 = RGB2YCbCr(image1)
            # Y2, Cb2, Cr2 = RGB2YCbCr(image2)

            # Y3, _, _ = RGB2YCbCr(label)
            # label = Y3
            # label = torch.max(image1, image2)
            # label = 0.7 * torch.max(image1, image2) + 0.3 * torch.min(image1, image2)
            # label = 0.7 * torch.max(Y1, Y2) + 0.3 * torch.min(Y1, Y2)
            # label = torch.max(Y1, Y2)
            # mask_image1 = label.to(device)  # 保留前景
            # mask_image2 = torch.ones_like(mask_image1)
            # mask_image2 = mask_image2 - mask_image1  # 保留背景
            # t = DFF.diffusion.sample_timesteps(image1.shape[0]).to(device)
            #
            # predicted_noise = net(image1, image2, t)
            # x_init, predicted_noisel, noise = net(image1, image2)
            # fusion_init = net(image1, image2)
            # loss = mse(noise, predicted_noise) + mse(x_init,label)
            # image1_out = net(image1)
            # image2_out = net(image2)
            image_out = net(image1, image2)
            # image_out = net(Y1, Y2)
            # loss = mse(mask_image1, image1_out) + mse(mask_image2, image2_out)
            # ssim = criterion(image1_out, label) + criterion(image2_out, label)
            # l1 = L1(image1_out, label) + L1(image2_out, label)
            # ssim = criterion(image1_out, image1) + criterion(image2_out, image2)
            # l1 = L1(image1_out, image1) + L1(image2_out, image2)
            #
            # loss = 0.8 * ssim + 0.2 * l1
            # loss = mse(image1, image1_out) + mse(image2, image2_out)
            # loss = mse(image1_out, label) + mse(image2_out, label)
            # loss = loss2 + l1 + ssim
            # label = torch.max(image1, image2)
            # loss = mse(image_out, label) + 0.8 * criterion(image_out, label) + 0.2 * L1(image_out, label)
            # loss = criterion(image_out, label) + L1(image_out, label)

            # loss1 = mse(image_out, label)
            loss2 = criterion(image_out, label)
            loss3 = L1(image_out, label)
            # loss =loss2
            #
            # # loss = 0.7 * loss1 + 0.2 * loss2 + 0.1 * loss3
            # loss4 = compute_loss(image_out, label)
            # loss5 = smoothL1(image_out, label)
            # # loss = loss1 + loss2 + loss3
            # # loss = loss2 + loss3
            # # loss = 10 * loss1 + loss2 + 2 * loss3 + 0.1 * loss4
            # # loss = 100 * loss1 + 10 * loss2 + loss3 + 0.1 * loss4
            # # loss = 100 * loss1 + loss2 + loss3 + 0.01 * loss4
            # # loss = 100 * loss1 + loss2 + 0.01 * loss4
            # # loss = loss3
            #
            # l1, l3, l4 = adaptive_coefficient(loss2, loss1, loss3, loss4)  # 以loss2为基准
            # loss = l1 * loss1 + loss2 + l3 * loss3 + l4 * loss4

            # loss = smoothL1(image_out, label)
            # l2, l4 = adaptive_coefficient2(loss5, loss2, loss4)
            # loss = loss5 + l2 * loss2 + l4 * loss4
            # loss = loss5

            # l2, l3, l4 = adaptive_coefficient(loss1, loss2, loss3, loss4)  # 以loss2为基准
            # loss = loss1 + l2 * loss2 + l3 * loss3 + l2 * loss4
            # l1, l2, l3 = adaptive_coefficient(loss4, loss1, loss2, loss3)
            # loss = l1 * loss1 + l2 * loss2 + l3 * loss3 + loss4
            l3 = adaptive_coefficient3(loss2, loss3)
            loss = loss2 + l3 * loss3
            # l2, l3 = adaptive_coefficient2(loss1, loss2, loss3)
            # loss = loss1 + l2 * loss2 + l3 * loss3
            # l4, l5 = adaptive_coefficient2(loss2, loss4, loss5)
            # loss = loss2 + l4 * loss4 + l5 * loss5
            # loss= loss2+loss4+loss5
            # loss = loss2 + loss3
            # l2 = adaptive_coefficient3(loss1, loss2)
            # loss = loss1 + l2 * loss2
            # loss = loss3

            # print(l2)
            # print(l3)
            # print(l4)
            # print(loss1)
            # print(loss2)
            # print(loss3)
            # print(loss4)

            # gradient_loss = compute_loss(fusion_init, label)
            # ssim = criterion(fusion_init, label)
            # l1 = L1(fusion_init, label)
            # loss = 0.1 * gradient_loss + 0.7 * ssim + 0.2 * l1
            # loss = ssim + l1 + loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            if (i + 1) % 8 == 0:
                # print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
                #     epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))
                # print(
                #     "[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f train loss1: %8f train loss2: %8f train loss3: %8f train loss4: %8f " % (
                #         epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item(), loss1.item(), loss2.item(),
                #         loss3.item(), loss4.item()))
                print(
                    "[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
                        epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))
        writer1.add_scalar('训练损失', running_loss, epoch)
        # if epoch % 2 == 0:
        #     check_point = {'state_dict': net.state_dict(),
        #                    'optimizer': optimizer.state_dict(),
        #                    'scheduler': scheduler.state_dict(),
        #                    'epoch': epoch}
        #     checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
        #     torch.save(check_point, checkpoint_path)

        check_point = {'state_dict': net.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(),
                       'epoch': epoch}
        checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
        torch.save(check_point, checkpoint_path)
        # if epoch % 4 == 0:
        #     check_point = {'net': net.state_dict(),
        #                    'optimizer': optimizer.state_dict(),
        #                    'scheduler': scheduler.state_dict(),
        #                    'epoch': epoch}
        #     checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
        #     torch.save(check_point, checkpoint_path)

        # sampled_images = net.diffusion.sample(net.unet, image1, image2, n=image1.shape[0])  # n=batch_size
        # x = x_init + sampled_images
        # x = x_init
        # x = (image1 + image2)/2 + sampled_images
        # x = sampled_images
        # x = image1
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # sampled_images = x
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

        # torch.save(net.state_dict(), model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss))
        running_loss = 0.0
        scheduler.step()
        t2 = time.time()
        print(t2 - t1)

    print('-------------Congratulations! Training Done!!!-------------')


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    # parser.add_argument('--run_name', type=str, default='DDPM_fusion5')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    # parser.add_argument('--dataset_dir', type=str, default="D:\DeepLearing\set\set-m")
    parser.add_argument('--dataset_dir', type=str, default=r'F:/train_dataset/Harvard')
    parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_pretrain19-3/')
    return parser.parse_args()  # 后调用parse_args()方法进行解析，解析成功之后即可使用


if __name__ == '__main__':
    training_setup_seed(1)
    args = parse_args()
    transforms_ = [transforms.Resize((64, 64)),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    # transforms_ = [transforms.Resize((256, 256)),
    #                # transforms.RandomHorizontalFlip(p=0.6),
    #                transforms.ToTensor()]
    train_set = Harvard(dataset_dir=args.dataset_dir, transforms_=transforms_)
    # train_set = CVC14(dataset_dir=args.dataset_dir, transforms_=transforms_)
    dataset_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train(dataset_dataloader, args)
