# ourlytro0917 ourmffw0917 ourwhu0917
import torch
# from data_loader import DataTest
# from data_loader_mfiwhu import DataTest
# from data_loader_histological import DataTest
from data_loader2 import DataTest
# from data_loader_tno import DataTest
# from data_loader_vi import DataTest
# from data_loader_ex import DataTest
# from data_loader_m import DataTest
# from data_loader_cell import DataTest
import argparse
from modules39 import DFF39
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from utils import save_images, test_save_images2, RGB2YCbCr, test_save_images


def test(test_dataloader, args):
    device = args.device

    # set_dir = args.testData_dir
    # set_list = os.listdir(set_dir)
    #获取文件名，去掉后缀
    file_list = os.listdir(args.testData_dir)
    temp_dir = os.listdir(os.path.join(args.testData_dir, file_list[0]))
    set_list = []
    for i in temp_dir:
        portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀

        set_list.append(portion[0])
    print(set_list)

    i = 0

    net = DFF39(args).to(device)
    # diffusion = DiffusionAndFusion37(img_size_h=args.image_size_h, img_size_w=args.image_size_w, device=device)

    checkpoint = torch.load(args.saveModel_dir)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    # net.load_state_dict(checkpoint['net'])
    # net.load_state_dict(torch.load(args.saveModel_dir))

    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain17\epoch_32_loss_26.353735.pth'
    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain17\epoch_8_loss_54.893467.pth'
    pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain19\epoch_88_loss_4.852962.pth'
    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain19-2\epoch_43_loss_0.414543.pth'
    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain19\epoch_164_loss_3.525550.pth'
    pretrained_net = torch.load(pretrained_checkpoint_path)
    # net.load_state_dict(pretrained_net['state_dict'], strict=False)
    net.load_state_dict(pretrained_net['state_dict'], strict=False)
    # model_dict = net.state_dict()
    # state_dict = {k: v for k, v in pretrained_net.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # net.load_state_dict(model_dict)
    print("加载预训练模型成功")

    net.eval()
    print(net)
    t1 = time.time()
    for i_test, (image1, image2) in enumerate(test_dataloader):
        image1 = image1.to(device)
        image2 = image2.to(device)

        x_init = net.first_stage_model(image1, image2)

        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, image2, image1, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, image2, image1, n=image1.shape[0])
        x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, x_init, image1, image2, n=image1.shape[0])
        x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, x_init, image2, image1, n=image1.shape[0])
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        EPSILON = 1e-10
        mask1 = torch.exp(image1) / (torch.exp(image2) + torch.exp(image1) + EPSILON)
        mask2 = torch.exp(image2) / (torch.exp(image1) + torch.exp(image2) + EPSILON)
        # # # # # # print(mask1)
        # # # # # # # print(mask2)
        sampled_images1 = mask1 * sampled_images1
        # # # # sampled_images1 = mask1 * image1
        # # # # #
        sampled_images2 = mask2 * sampled_images2
        sampled_images = sampled_images1 + sampled_images2

        out = sampled_images
        # out = x_noise
        # out = x_init
        out = torch.squeeze(out)
        # sampled_images1 = torch.squeeze(sampled_images1)
        # sampled_images2 = torch.squeeze(sampled_images2)
        # test_save_images(sampled_images1, "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion29/" + set_list[i] + "_1.png")
        # test_save_images(sampled_images2, "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion29/" + set_list[i] + "_2.png")

        #
        # out = sampled_images
        # out = image2 - x_init
        # out = (out.clamp(-1, 1) + 1) / 2
        # out = (out * 255).type(torch.uint8)
        # out = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)
        # out_image = (out * 255).type(torch.uint8)
        # out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))
        # save_images(sampled_images, os.path.join("test_results_DDPM_fusion6", f"{i}.jpg"))
        # out_image = out_image.resize((520, 520))  ####重设图像大小

        # out_image.save("result/UFA/" + "color_lytro_" + str(i_test + 1).zfill(2) + ".png")
        # print("color_lytro_" + str(i_test + 1).zfill(2) + ".png")

        # out_image.save("result/our-m/" + set_list[i] + ".png")
        # print("result/our-m/" + set_list[i] + ".png")
        # save_images(out_image, "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion24/" + set_list[i] + ".png")
        test_save_images(out, "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion39-2/" + set_list[i] + ".png")
        # out_image.save("D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion24/" + set_list[i] + ".png")
        print("D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion39-2/" + set_list[i] + ".png")
        i = i + 1
    t2 = time.time()
    print(t2 - t1)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_size_h', type=int, default=520)
    # parser.add_argument('--image_size_w', type=int, default=520)
    parser.add_argument('--image_size_h', type=int, default=520)
    parser.add_argument('--image_size_w', type=int, default=520)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--testData_dir', type=str,
    #                     default="D:/DeepLearing/pythonProject/code/DataSet/TestData/MF/lytro")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\MFI_WHU_select")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\TNO_select")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\Roadscene_select")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\MSRS_select")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\Histological_select")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\lytro2")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\MFFW_select2")
    parser.add_argument('--testData_dir', type=str,
                        default=r"F:\test_dataset\MFI_WHU_select2")

    # parser.add_argument('--testData_dir', type=str, default="D:\DeepLearing\set\set-ex-test")
    # parser.add_argument('--testData_dir', type=str,
    #                     default="D:\DeepLearing\set\set-m-test4")
    # parser.add_argument('--testData_dir', type=str,
    #                     default="D:\DeepLearing\set\set-m-test3")
    # parser.add_argument('--testData_dir', type=str, default="D:\DeepLearing\set\set-cell-test")
    # parser.add_argument('--testData_dir', type=str, default="D:\DeepLearing\set\set-vi-test3")
    # parser.add_argument('--testData_dir', type=str, default="D:/DeepLearing/set/MFFW")
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion38/epoch_152_loss_1.434056.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion38-2/epoch_104_loss_0.803233.pth')
    parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion39/epoch_144_loss_0.514635.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion38-2/epoch_192_loss_0.303291.pth')
    parser.add_argument('--result', type=str, default='result')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # transforms_ = [transforms.Resize(size=(args.image_size_h, args.image_size_w)), transforms.ToTensor()]
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir, transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)
