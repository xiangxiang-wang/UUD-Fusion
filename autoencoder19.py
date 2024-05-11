import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder19(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Autoencoder19, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, img1, img2):
        z1 = self.encoder(img1)

        z2 = self.encoder(img2)

        z = torch.cat([z1, z2], dim=1)

        output = self.decoder(z)
        return output


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.extract_features = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                              # nn.GroupNorm(1, 64),
                                              nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                              # nn.ReLU(),
                                              # DownSample(64),
                                              nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2, bias=True),
                                              # nn.GroupNorm(1, 96),
                                              nn.LeakyReLU(negative_slope=1e-2, inplace=True),

                                              # nn.ReLU(),
                                              # DownSample(96),
                                              nn.Conv2d(96, 128, kernel_size=7, stride=1, padding=3, bias=True),
                                              # nn.GroupNorm(1, 128),
                                              nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                              # nn.ReLU(),
                                              # DownSample(128)
                                              )
        # self.conv_out = nn.Conv2d(128, z_channels, 1, stride=1)
        # self.dropout = torch.nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, img: torch.Tensor):
        x = self.extract_features(img)
        # x = self.dropout(x)
        # x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.reconstruct = nn.Sequential(nn.Conv2d(256, 96, kernel_size=7, stride=1, padding=3, bias=True),
                                         # nn.GroupNorm(1, 96),
                                         nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                         # nn.ReLU(),
                                         # UpSample(),
                                         nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=2, bias=True),
                                         # nn.GroupNorm(1, 64),
                                         nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                         # nn.ReLU(),
                                         # UpSample(),
                                         nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                         # nn.GroupNorm(1, 32),
                                         nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                         # nn.ReLU(),
                                         # nn.Dropout2d(p=0.3, inplace=False),
                                         # UpSample(),
                                         nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                         # nn.GroupNorm(1, out_channels),
                                         nn.Sigmoid()
                                         )
        # self.conv_in = nn.Conv2d(2 * z_channels, 256, 1, stride=1)
        # self.dropout = torch.nn.Dropout2d(p=0.3, inplace=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # h = self.conv_in(z)
        img = self.reconstruct(h)
        # img = img * self.sigmoid(img)

        return img

# class Encoder(nn.Module):
#     def __init__(self, in_channels):
#         super(Encoder, self).__init__()
#         self.extract_features = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
#                                               nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                               # DownSample(64),
#                                               nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
#                                               nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                               # DownSample(96),
#                                               nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True),
#                                               nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                               # DownSample(128)
#                                               )
#         # self.conv_out = nn.Conv2d(128, z_channels, 1, stride=1)
#         # self.dropout = torch.nn.Dropout2d(p=0.5, inplace=False)
#
#     def forward(self, img: torch.Tensor):
#         x = self.extract_features(img)
#         # x = self.dropout(x)
#         # x = self.conv_out(x)
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, out_channels):
#         super(Decoder, self).__init__()
#         self.reconstruct = nn.Sequential(nn.Conv2d(256, 128, kernel_size=7, stride=1, padding=3, bias=True),
#                                          nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                          # UpSample(),
#                                          nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True),
#                                          nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                          # UpSample(),
#                                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
#                                          nn.LeakyReLU(negative_slope=1e-2, inplace=True),
#                                          # nn.Dropout2d(p=0.5, inplace=False),
#                                          # UpSample(),
#                                          nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
#                                          nn.LeakyReLU(negative_slope=1e-2, inplace=True))
#         # self.conv_in = nn.Conv2d(2 * z_channels, 256, 1, stride=1)
#         # self.dropout = torch.nn.Dropout2d(p=0.3, inplace=False)
#
#     def forward(self, h):
#         # h = self.conv_in(z)
#         img = self.reconstruct(h)
#         return img
#
#
#
# # class UpSample(nn.Module):
# #     """
# #     ## Up-sampling layer
# #     """
# #
# #     def __init__(self):
# #         """
# #         :param channels: is the number of channels
# #         """
# #         super().__init__()
# #         # $3 \times 3$ convolution mapping
# #         # self.conv = nn.Conv2d(channels, channels, 3, padding=1)
# #
# #     def forward(self, x: torch.Tensor):
# #         """
# #         :param x: is the input feature map with shape `[batch_size, channels, height, width]`
# #         """
# #         # Up-sample by a factor of $2$
# #         x = F.interpolate(x, scale_factor=2.0, mode="nearest")
# #         # Apply convolution
# #         # return self.conv(x)
# #
# #         return x
# #
# #
# # class DownSample(nn.Module):
# #     """
# #     ## Down-sampling layer
# #     """
# #
# #     def __init__(self, channels: int):
# #         """
# #         :param channels: is the number of channels
# #         """
# #         super().__init__()
# #         # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
# #         # self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)
# #         self.down = nn.MaxPool2d(2)
# #
# #     def forward(self, x: torch.Tensor):
# #         """
# #         :param x: is the input feature map with shape `[batch_size, channels, height, width]`
# #         """
# #         # # Add padding
# #         # x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
# #         # # Apply convolution
# #         # return self.conv(x)
# #         x = self.down(x)
# #         return x