import torch.nn as nn
from autoencoder19 import Autoencoder19


# class Autoencoder14(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.first_stage_model = Autoencoder6(in_channels=3, out_channels=3, z_channels=4)
#
#     def forward(self, image1, image2):
#         x = self.first_stage_model(image1, image2)
#         return x

# class Pretrain15(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.first_stage_model = Autoencoder7(in_channels=1, out_channels=1, z_channels=4)
#
#     def forward(self, image1, image2):
#         x = self.first_stage_model(image1, image2)
#         return x

class Pretrain19(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_stage_model = Autoencoder19(in_channels=3, out_channels=3)

    def forward(self, image1, image2):
        x = self.first_stage_model(image1, image2)
        return x