# data loader
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random


# Multi-Focus-Dataset：DUTS
class MultiFocusDatasetDUTS(Dataset):
    def __init__(self, dataset_dir, transforms_, rgb=True):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(self.dataset_dir)
        self.transform = transforms.Compose(transforms_)
        self.rgb = rgb

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        temp_dir = os.listdir(os.path.join(self.dataset_dir, self.file_list[item]))
        temp_item = random.randint(0, 3)
        child_dir = temp_dir[temp_item]
        if self.rgb:
            img1 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[item] + "_1.jpg")
            img2 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[item] + "_2.jpg")
            label = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[
                    item] + "_ground.jpg")
        else:
            img1 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[
                    item] + "_1.jpg").convert('L')
            img2 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[
                    item] + "_2.jpg").convert('L')
            label = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + '/' + child_dir + '/' + self.file_list[
                    item] + "_ground.jpg").convert('L')
        # img1 = img1.resize((64, 64), Image.BICUBIC)
        # img2 = img2.resize((64, 64), Image.BICUBIC)
        # label = label.resize((64, 64), Image.BICUBIC)
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = self.transform(label)
        return img1, img2, label


class DataTest(Dataset):
    def __init__(self, testData_dir, transforms_):
        self.testData_dir = testData_dir  # 测试文件路径
        self.file_list = os.listdir(testData_dir)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        self.transform = transforms.Compose(transforms_)  # 串联多个transform操作

    def __getitem__(self, idx):
        image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg")
        image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg")
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1, image2

    def __len__(self):
        return len(self.file_list)

