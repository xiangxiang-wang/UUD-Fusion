# SCIE oursice0913 oursice2-0917 oursice3-0917
import torch
# from data_loader import DataTest
# from data_loader_mfiwhu import DataTest
from data_loader2 import DataTest
# from data_loader_mffw import DataTest
# from data_loader_SICE_MEF1 import DataTest
# from data_loader_harvard import DataTest
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
    # 获取文件名，去掉后缀
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
    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain19\epoch_164_loss_3.525550.pth'
    # pretrained_checkpoint_path = r'E:\saveModel_dir_pretrain19-2\epoch_76_loss_0.666261.pth'
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
        # Y1, Cb1, Cr1 = RGB2YCbCr(image1)
        # Y2, Cb2, Cr2 = RGB2YCbCr(image2)
        # out = net(image1, image2)
        # out, _ = net(image1, image2)
        # images = image1
        # x_init = net.fusion(image1, image2)
        # sampled_images = diffusion.sample2(net.unet, image1, image2, n=image1.shape[0])  # n=batch_size
        # x_init = (image1+image2)/2
        # x_res = net.autoencoder_encode(x_init)
        # x1 = net.autoencoder_encode(image1)
        # x2 = net.autoencoder_encode(image2)
        # x_init = torch.max(x1, x2)
        # x0 = (image1+image2)/2
        # res1 = x0 - image1
        # res2 = x0 - image2
        # x_res = torch.min(res1, res2)
        # x_init = net.autoencoder_encode(x_res)
        # x1 = net.first_stage_model.encoder(image1)
        # x2 = net.first_stage_model.encoder(image2)
        # x_init = (x1+x2)/2
        # x_init = (image1 + image2)/2
        # x_init = net.autoencoder_encode(x0)
        # x_init = torch.max(x1,x2)
        # x_init = x1
        # x_init = torch.cat([x1, x2], dim=1)
        # x_init = torch.max(x1, x2)
        # x_init = net.first_stage_model.decoder(x_init)
        # x_res1 = Y1 - x_init
        # x_res2 = Y2 - x_init
        # x_res1 = self.first_stage_model.encoder(x_res1)
        # x_res2 = self.first_stage_model.encoder(x_res2)
        # x_res = torch.max(x_res1, x_res2)
        # x_res = torch.cat([x_res1,x_res2],dim=1)

        x_init = net.first_stage_model(image1, image2)

        # x_init = net.first_stage_model.decoder(x_init)
        # x_res1 = image1 - x_init
        # x_res2 = image2 - x_init
        # x_res = torch.max(x_res1, x_res2)
        # x_res = (x_res1+x_res2)/2
        # x_res = (image1 + image2) / 2 - x_init
        # x0 = (Y1+Y2)/2
        # x0 = net.first_stage_model.encoder(x0)
        # x_res = x_init - x0
        # print(x_res.shape)
        # print(x_init.shape)

        # sampled_images = diffusion.sample2(net.unet, Y1, Y2, n=image1.shape[0])

        # x_noise, sampled_images = net.diffusion.sample_from_blur(net.unet, x_init, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur3(net.unet, x_init,image1,image2, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur(net.unet, x_init, n=image1.shape[0])
        # sampled_images1=image1
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur9(net.unet,  x_init, image1, image2, n=image1.shape[0])
        # x_noise, sampled_images_ = net.diffusion.sample_from_blur9(net.unet, x_init, image1, image2, n=image1.shape[0])
        # sampled_images_ = x_init
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, image2, image1, n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur9(net.unet, x_init, image1, image2, n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur16(net.unet, x_init, image1, image2,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur16(net.unet, x_init, image2, image1,
        #                                                             n=image1.shape[0])
        x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, x_init, image1, image2, n=image1.shape[0])
        x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, x_init, image2, image1, n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur16(net.unet, x_init, image1, image2, n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur16(net.unet, x_init, image2, image1, n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, sampled_images2, sampled_images1, n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, x_init, sampled_images_,  sampled_images_, n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, sampled_images_,image1, image2,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet,  sampled_images_, image2, image1,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, sampled_images_, image1, image2,
        #                                                             n=image1.shape[0])
        # x_init = (image1 + image2)/2

        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet,  image1, image1, image2,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, sampled_images_, image2, image1,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, sampled_images1, image1, image2,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images = net.diffusion.sample_from_blur14(net.unet, image2, image2, image1,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, image1, image1, image2,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet,image2, image2, image1,
        #                                                             n=image1.shape[0])
        # sampled_images1 = image1
        # sampled_images1 = sampled_images1 + image2
        # sampled_images2 = sampled_images2 + image1
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, sampled_images1,image1, image2, n=image1.shape[0])


        # sampled_images2 = image1
        # sampled_images = sampled_images2 + image2
        # x_noise, sampled_images = net.diffusion.sample_from_blur9(net.unet, x_init, sampled_images1, sampled_images2, n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, image1, image1, sampled_images2, n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, sampled_images1,image2, image1,
        #                                                              n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur14(net.unet, image1, image1, sampled_images1,
        #                                                             n=image1.shape[0])
        # x_noise, sampled_images1 = net.diffusion.sample_from_blur16(net.unet, x_i, sampled_images2, sampled_images1, n=image1.shape[0])
        # x_noise, sampled_images2 = net.diffusion.sample_from_blur14(net.unet, image2, image2, image1, n=image1.shape[0])

        # x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_, sampled_images1 = net.diffusion.sample_from_blur17(net.unet, x_init, image1, image2, n=image1.shape[0])
        # x0_bar_, x0_bar2_, x0_bar3_, img_, x_t_bar3_, sampled_images2 = net.diffusion.sample_from_blur17(net.unet, x_init, image2, image1, n=image1.shape[0])
        # for j in range(len(x0_bar_)):
        #     test_save_images(torch.squeeze(x0_bar_[j]),
        #                      r"D:\BaiduSyncdisk\diffusion-fusion\test_results_DDPM_fusion39-2\x0_bar_/" + str(j) + "_x0_bar_.png")
        # for j in range(len(x0_bar2_)):
        #     test_save_images(torch.squeeze(x0_bar2_[j]),
        #                      r"D:\BaiduSyncdisk\diffusion-fusion\test_results_DDPM_fusion39-2\x0_bar2_/" + str(j) + "_x0_bar2_.png")
        # for j in range(len(x0_bar3_)):
        #     test_save_images(torch.squeeze(x0_bar3_[j]),
        #                      r"D:\BaiduSyncdisk\diffusion-fusion\test_results_DDPM_fusion39-2\x0_bar3_/" + str(j) + "_x0_bar3_.png")
        # for j in range(len(img_)):
        #     test_save_images(torch.squeeze(img_[j]),
        #                      r"D:\BaiduSyncdisk\diffusion-fusion\test_results_DDPM_fusion39-2\img_/" + str(j) + "_img_.png")
        # for j in range(len(x_t_bar3_)):
        #     test_save_images(torch.squeeze(x_t_bar3_[j]),
        #                      r"D:\BaiduSyncdisk\diffusion-fusion\test_results_DDPM_fusion39-2\x_t_bar3_/" + str(j) + "_x_t_bar3_.png")

        # sampled_images = torch.max(sampled_images1, sampled_images2)
        # sampled_images1 = (sampled_images1 + sampled_images_)/2
        # sampled_images2 = (sampled_images2 + sampled_images_)/2
        # sampled_images1 = torch.max(sampled_images1, image1)
        # sampled_images2 = torch.max(sampled_images2,image2)
        # sampled_images2 = torch.max(sampled_images2, image1)
        # sampled_images1 = (sampled_images1 + image2)
        # sampled_images2 = (sampled_images2 + image1)
        # sampled_images1 =image2
        # sampled_images2 = torch.max(sampled_images2, image2)
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        # _, sampled_images1 = net.ch_att(image1, sampled_images1)
        # _, sampled_images2 = net.ch_att(image2, sampled_images2)
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        # sampled_images2 = torch.max(sampled_images2, image2)
        # sampled_images2 = (sampled_images2 + image1)
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        # mask1,mask2 = net.ch_att(sampled_images1, sampled_images2)
        # sampled_images1 = torch.max(sampled_images1, image1)
        # sampled_images2 =torch.max(sampled_images2, image2)
        # sampled_images = (sampled_images1 + sampled_images2)
        # sampled_images1 = x_init
        # sampled_images2 = sampled_images2 + image2
        # sampled_images1, sampled_images2 = net.ch_att(sampled_images1, sampled_images2)
        # EPSILON = 1e-10
        # # mask1 = torch.exp(sampled_images1) / (torch.exp(sampled_images2) + torch.exp(sampled_images1)+EPSILON)
        # # mask2 = torch.exp(sampled_images2) / (torch.exp(sampled_images1) + torch.exp(sampled_images2)+EPSILON)
        # mask1 = torch.max(sampled_images1, sampled_images2)
        # # mask2 = torch.min(sampled_images1, sampled_images2)
        EPSILON = 1e-10
        mask1 = torch.exp(image1) / (torch.exp(image2) + torch.exp(image1) + EPSILON)
        mask2 = torch.exp(image2) / (torch.exp(image1) + torch.exp(image2) + EPSILON)
        # # # # # # print(mask1)
        # # # # # # # print(mask2)
        sampled_images1 = mask1 * sampled_images1
        # # # # sampled_images1 = mask1 * image1
        # # # # #
        sampled_images2 = mask2 * sampled_images2
        # # # sampled_images1 = x_init
        # sampled_images = (sampled_images1 + sampled_images2)
        # # sampled_images = sampled_images + sampled_images_
        # sampled_images = torch.max(image1, sampled_images)
        # sampled_images = sampled_images + image1
        # sampled_images1 = image2
        # sampled_images = torch.max(x_init, sampled_images)
        sampled_images = torch.max(sampled_images1, sampled_images2)
        # sampled_images = torch.min(sampled_images, image1)
        # sampled_images = (sampled_images + image1) / 2
        # sampled_images = sampled_images1
        # sampled_images1 = torch.max(image1, sampled_images1)
        # sampled_images = torch.max(x_init, sampled_images)
        # sampled_images = torch.max(image2, sampled_images)
        # sampled_images = torch.min(sampled_images_, sampled_images)
        # sampled_images = torch.max(image1, sampled_images)
        # sampled_images = torch.max(image2, sampled_images)
        # mask2 = torch.min(sampled_images1, sampled_images2)
        # sampled_images = net.first_stage_model(sampled_images1, sampled_images2)
        # sampled_images1 = image1
        # sampled_images = torch.max(sampled_images1, sampled_images2)
        # sampled_images1 = image1
        # sampled_images = sampled_images2
        # sampled_images = sampled_images2
        # mask = img_[len(img_)-2].clone()
        # sampled_images1 = 2 * mask1 * image1
        # sampled_images2 = 2 * mask2 * image2
        # sampled_images = net.first_stage_model(sampled_images1, sampled_images2)
        # sampled_images = (sampled_images + image2) / 2
        # sampled_images = net.first_stage_model(sampled_images, image2)
        # sampled_images1 = torch.min(sampled_images1, image1)
        # sampled_images2 = torch.min(sampled_images2, image2)
        # sampled_images = torch.max(sampled_images1, sampled_images2)
        # sampled_images = torch.max(sampled_images, image2)
        # sampled_images = (mask*sampled_images2 + (1-mask)*sampled_images1)/2
        # x_noise, sampled_images, x_noise1, x_noise2, x0 = net.diffusion.sample_from_blur10(net.unet, x_init, image1, image2, n=image1.shape[0])
        # print(x_noise2[0].size())
        # print(len(x_noise1))
        # print(len(x_noise2))
        # print(len(x0))

        # for i in range(50):
        #     test_save_images(torch.squeeze(x_noise1[i]),
        #                      "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion38-2/x_noise/" + str(i) + "_x_noise1.png")
        # for i in range(51):
        #     test_save_images(torch.squeeze(x0[i]),
        #                      "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion38-2/x0/" + str(i) + "_x0.png")
        # for i in range(49):
        #     test_save_images(torch.squeeze(x_noise2[i]),
        #                      "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion38-2/x_noise/" + str(i) + "_x_noise2.png")
        #
        # print("中间图像已生成")

        # x_noise, sampled_images = net.diffusion.sample_from_blur4(net.unet, x_res, n=image1.shape[0])

        # sampled_images = diffusion.sample5(net.unet, x_init, Y1, Y2, n=image1.shape[0])  # n=batch_size
        # sampled_images = diffusion.sample2(net.unet, x_res, Y1, Y2, n=image1.shape[0])  # n=batch_size
        # sampled_images = x_init + sampled_images
        # sampled_images = diffusion.sample2(net.unet, x_init, x1, x2, n=image1.shape[0])  # n=batch_size

        # sampled_images = net.first_stage_model.decoder(sampled_images)

        # sampled_images1 = net.autoencoder_decode1(sampled_images)
        # sampled_images2 = net.autoencoder_decode2(sampled_images)
        # sampled_images = (sampled_images1 + sampled_images2) / 2
        # sampled_images = net.autoencoder_decode(sampled_images)
        # x = x_init + sampled_images
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # sampled_images = x
        # sampled_images = x_init
        # sampled_images = torch.cat([sampled_images, sampled_images], dim=1)
        # sampled_images = net.initial_result(sampled_images)
        # x1 = net.extract_features1(image1)
        # x2 = net.extract_features2(image2)
        # x = torch.cat([x1, x2], dim=1)
        # x_init = net.initial_result(x)
        # sampled_images = x_init + sampled_images
        # out = x_init + sampled_images
        # out = x0 + sampled_images
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
        # i=0
        test_save_images(out, "D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion39-2/" + set_list[i] + ".png")
        # out_image.save("D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion24/" + set_list[i] + ".png")
        print("D:/BaiduSyncdisk/diffusion-fusion/test_results_DDPM_fusion39-2/" + set_list[i] + ".png")
        # break
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
    #                     default=r"F:\test_dataset\SICE_MEF1_select2")
    # parser.add_argument('--testData_dir', type=str,
    #                     default=r"F:\test_dataset\SICE_MEF2_select2")
    parser.add_argument('--testData_dir', type=str,
                        default=r"F:\test_dataset\SICE_MEF3_select2")

    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion38/epoch_152_loss_1.434056.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion38-2/epoch_104_loss_0.803233.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion39/epoch_144_loss_0.514635.pth')
    parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion39-3/epoch_46_loss_1.127614.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:/saveModel_dir_fusion39-3/epoch_90_loss_2.631842.pth')
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
