import os
from math import log10

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def mask_attention(image1,image2):
    image1_0 = image1[:, 0, :, :]
    image1_1 = image1[:, 1, :, :]
    image1_2 = image1[:, 2, :, :]
    image1_0 = torch.unsqueeze(image1_0, dim=1)
    image1_1 = torch.unsqueeze(image1_1, dim=1)
    image1_2 = torch.unsqueeze(image1_2, dim=1)

    image1_c1 = torch.cat([image1_0, image1_0, image1_0], dim=1)
    image_temp_c1 = image2 - image1_c1
    # image_temp_c1 = x_init - image1_c1

    image_temp_c1 = image_temp_c1.clamp_(0, 1)

    image1_c2 = torch.cat([image1_1, image1_1, image1_1], dim=1)
    image_temp_c2 = image2 - image1_c2
    # image_temp_c2 = x_init - image1_c2
    image_temp_c2 = image_temp_c2.clamp_(0, 1)

    image1_c3 = torch.cat([image1_2, image1_2, image1_2], dim=1)
    image_temp_c3 = image2 - image1_c3
    # image_temp_c3 = x_init - image1_c3
    image_temp_c3 = image_temp_c3.clamp_(0, 1)

    out_premask = torch.min(image_temp_c1, image_temp_c2)
    out_premask = torch.min(out_premask, image_temp_c3)
    # out_premask = image_temp_c1+image_temp_c2+image_temp_c3
    # out_premask = torch.max(image_temp_c1, image_temp_c2)
    # out_premask = torch.max(out_premask, image_temp_c3)
    out_premask = torch.abs(out_premask)
    return out_premask

# # 调用语句
# # save_image(out, path, nrow=4, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
# def save_images2(images, path, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
#     grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, pad_value=pad_value,
#                                        normalize=normalize, range=range, scale_each=scale_each)
#     ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)

def save_images2(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
def save_images3(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)



def test_save_images(image, path):
    ndarr = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def test_save_images2(image, path):
    ndarr = image.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def test_save_images3(image, path):
    ndarr = image.mul_(255).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    # dataset,它的结构就是[(img_data,class_id),(img_data,class_id),…],
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # 构建可迭代的数据装载器, 我们在训练的时候，每一个for循环，每一次iteration，就是从DataLoader中获取一个batch_size大小的数据的。
    #
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def compute_loss(fusion, img_cat):
    loss2 = structure_loss(fusion, img_cat)

    return loss2


#
# def compute_loss(fusion, img_1, img_2, put_type='mean'):
#     loss1 = intensity_loss(fusion, img_1, img_2, put_type)
#
#     return loss1

# def create_putative(in1, in2, put_type):
#     if put_type == 'mean':
#         iput = (in1 + in2) / 2
#     elif put_type == 'left':
#         iput = in1
#     elif put_type == 'right':
#         iput = in2
#     else:
#         raise EOFError('No supported type!')
#
#     return iput
#
#
# def intensity_loss(fusion, img_1, img_2, put_type):
#     inp = create_putative(img_1, img_2, put_type)
#
#     # L2 norm
#     loss = torch.norm(fusion - inp, 2)
#
#     return loss


def gradient(x):
    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]  ##扩充 左右上下
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def create_structure(inputs):
    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W)  # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:, 0, :, :] = torch.sum(a_00, dim=1)
    structure[:, 1, :, :] = torch.sum(a_01, dim=1)
    structure[:, 2, :, :] = torch.sum(a_10, dim=1)
    structure[:, 3, :, :] = torch.sum(a_11, dim=1)

    return structure


def structure_loss(fusion, img_cat):
    st_fusion = create_structure(fusion)
    st_input = create_structure(img_cat)

    # Frobenius norm
    loss = torch.norm(st_fusion - st_input)

    return loss


#  以loss_r的数量级为基准
def adaptive_coefficient(loss_r, loss1, loss2, loss3):
    # quotient1 = loss_r // loss1
    # quotient2 = loss_r // loss2
    quotient1 = torch.div(loss_r, loss1, rounding_mode='floor')
    quotient2 = torch.div(loss_r, loss2, rounding_mode='floor')
    quotient3 = torch.div(loss_r, loss3, rounding_mode='floor')
    if quotient1 == 0:
        quotient1_decimal = torch.div(loss1, loss_r, rounding_mode='floor')
        order1 = int(log10(quotient1_decimal))
        l1 = 0.1 ** order1
    else:

        order1 = int(log10(quotient1))
        l1 = 10 ** order1

    if quotient2 == 0:
        quotient2_decimal = torch.div(loss2, loss_r, rounding_mode='floor')
        order2 = int(log10(quotient2_decimal))
        l2 = 0.1 ** order2
    else:
        order2 = int(log10(quotient2))
        l2 = 10 ** order2

    if quotient3 == 0:
        quotient3_decimal = torch.div(loss3, loss_r, rounding_mode='floor')
        order3 = int(log10(quotient3_decimal))
        l3 = 0.1 ** order3
    else:
        order3 = int(log10(quotient3))
        l3 = 10 ** order3

    return l1, l2, l3


#  以loss_r的数量级为基准
def adaptive_coefficient2(loss_r, loss1, loss2):
    # quotient1 = loss_r // loss1
    # quotient2 = loss_r // loss2
    quotient1 = torch.div(loss_r, loss1, rounding_mode='floor')
    quotient2 = torch.div(loss_r, loss2, rounding_mode='floor')

    if quotient1 == 0:
        quotient1_decimal = torch.div(loss1, loss_r, rounding_mode='floor')
        order1 = int(log10(quotient1_decimal))
        l1 = 0.1 ** order1
    else:

        order1 = int(log10(quotient1))
        l1 = 10 ** order1

    if quotient2 == 0:
        quotient2_decimal = torch.div(loss2, loss_r, rounding_mode='floor')
        order2 = int(log10(quotient2_decimal))
        l2 = 0.1 ** order2
    else:
        order2 = int(log10(quotient2))
        l2 = 10 ** order2

    return l1, l2


def adaptive_coefficient3(loss_r, loss1):
    # quotient1 = loss_r // loss1
    # quotient2 = loss_r // loss2
    quotient1 = torch.div(loss_r, loss1, rounding_mode='floor')

    if quotient1 == 0:
        quotient1_decimal = torch.div(loss1, loss_r, rounding_mode='floor')
        order1 = int(log10(quotient1_decimal))
        l1 = 0.1 ** order1
    else:

        order1 = int(log10(quotient1))
        l1 = 10 ** order1

    return l1


# #  以loss_r的数量级为基准
# def adaptive_coefficient(loss_r, loss1, loss2, loss3):
#     # quotient1 = loss_r // loss1
#     # quotient2 = loss_r // loss2
#     quotient1 = torch.div(loss_r, loss1, rounding_mode='floor')
#     quotient2 = torch.div(loss_r, loss2, rounding_mode='floor')
#     quotient3 = torch.div(loss_r, loss3, rounding_mode='floor')
#     if quotient1 != 0:
#         order1 = int(log10(quotient1))
#         l1 = 10 ** order1
#     else:
#         l1 = 1
#
#     if quotient2 != 0:
#         order2 = int(log10(quotient2))
#         l2 = 10 ** order2
#     else:
#         l2 = 1
#
#     if quotient3 != 0:
#         order3 = int(log10(quotient3))
#         l3 = 10 ** order3
#     else:
#         l3 = 1
#
#     return l1, l2, l3


# def YCrCb2RGB(input_im, device):
#     im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
#     # bchw-->bwhc-->bhwc-->bhw,c
#     mat = torch.tensor(
#         [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
#     ).to(device)
#     bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
#     temp = (im_flat + bias).mm(mat).to(device)  # 矩阵相乘
#     out = (
#         temp.reshape(
#             list(input_im.size())[0],
#             list(input_im.size())[2],
#             list(input_im.size())[3],
#             3,
#         )
#             .transpose(1, 3)
#             .transpose(2, 3)
#     )
#     return out
#
#
# def RGB2YCrCb(input_im, device):
#     # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
#     im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
#     # bchw-->bwhc-->bhwc-->bhw,c
#     R = im_flat[:, 0]
#     G = im_flat[:, 1]
#     B = im_flat[:, 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     Cr = (R - Y) * 0.713 + 0.5
#     Cb = (B - Y) * 0.564 + 0.5
#     Y = torch.unsqueeze(Y, 1)
#     Cr = torch.unsqueeze(Cr, 1)
#     Cb = torch.unsqueeze(Cb, 1)
#     temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
#     out = (
#         temp.reshape(
#             list(input_im.size())[0],
#             list(input_im.size())[2],
#             list(input_im.size())[3],
#             3,
#         )
#             .transpose(1, 3)
#             .transpose(2, 3)
#     )
#     # b,h,w,c-->b,c,w,h-->b,c,h,w
#     return out


def RGB2YCbCr(img_rgb):
    R = img_rgb[:, 0, :, :]
    G = img_rgb[:, 1, :, :]
    B = img_rgb[:, 2, :, :]
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 0.0625
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 0.5
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 0.5
    Y = torch.unsqueeze(Y, dim=1)
    Cb = torch.unsqueeze(Cb, dim=1)
    Cr = torch.unsqueeze(Cr, dim=1)
    return Y, Cb, Cr


def YCbCr2RGB(img_YCbCr):
    Y = img_YCbCr[:, 0, :, :]
    Cb = img_YCbCr[:, 1, :, :]
    Cr = img_YCbCr[:, 2, :, :]

    R = 1.164 * (Y - 0.0625) + 1.596 * (Cr - 0.5)
    G = 1.164 * (Y - 0.0625) - 0.392 * (Cb - 0.5) - 0.813 * (Cr - 0.5)
    B = 1.164 * (Y - 0.0625) + 2.017 * (Cb - 0.5)

    image_RGB = torch.cat([R, G, B], dim=1)
    return image_RGB


def imgYCbCrfusion(Y, Cb1, Cb2, Cr1, Cr2):
    b, c, h, w = Cb1.shape
    Cb = torch.ones_like(Cb1)
    Cr = torch.ones_like(Cr1)
    for k in range(h):
        for n in range(w):
            if torch.abs(Cb1[:, :, k, n] - 0.5) == 0 and torch.abs(Cb2[:, :, k, n] - 0.5) == 0:
                Cb[:, :, k, n] = 0.5
            else:
                middle_1 = Cb1[:, :, k, n] * torch.abs(Cb1[:, :, k, n] - 0.5) + Cb2[:, :, k, n] * torch.abs(
                    Cb2[:, :, k, n] - 0.5)
                middle_2 = torch.abs(Cb1[:, :, k, n] - 0.5) + torch.abs(Cb2[:, :, k, n] - 0.5)
                Cb[:, :, k, n] = middle_1 / middle_2

            if torch.abs(Cr1[:, :, k, n] - 0.5) == 0 and torch.abs(Cr2[:, :, k, n] - 0.5) == 0:
                Cr[:, :, k, n] = 0.5
            else:
                middle_3 = Cr1[:, :, k, n] * torch.abs(Cr1[:, :, k, n] - 0.5) + Cr2[:, :, k, n] * torch.abs(
                    Cr2[:, :, k, n] - 0.5)
                middle_4 = torch.abs(Cr1[:, :, k, n] - 0.5) + torch.abs(Cr2[:, :, k, n] - 0.5)
                Cr[:, :, k, n] = middle_3 / middle_4

    output = torch.cat([Y, Cb, Cr], dim=1)
    output = YCbCr2RGB(output)
    return output
