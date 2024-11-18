import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# 构建ForecastNet结构
class ForecastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, 3, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=(30, 30), stride=(3, 3), padding=14, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 构建ForecastSpaceNet结构
class ForecastSpaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 10, 3, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=(30, 30), stride=(3, 3), padding=14, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.image_list) - 3  # 每连续的4张图片作为一个数据

    def __getitem__(self, idx):
        # 加载四张图片
        img1 = Image.open(os.path.join(self.root_dir, self.image_list[idx]))
        img2 = Image.open(os.path.join(self.root_dir, self.image_list[idx + 1]))
        img3 = Image.open(os.path.join(self.root_dir, self.image_list[idx + 2]))
        img4 = Image.open(os.path.join(self.root_dir, self.image_list[idx + 3]))

        # 可选：应用图像转换
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        # 将前三张图片的灰度信息作为三通道的输入，第四张图片作为标签
        input_data = torch.cat((img1, img2, img3), dim=0)
        label = img4

        return input_data, label


# 自定义预处理方法
class ConvertGrayscale(torchvision.transforms.Lambda):
    def __init__(self, color_depth):
        """
        :param color_depth: 需要转换到的色深
        """
        super().__init__(self.convert_grayscale)
        self.color_depth = color_depth

    def convert_grayscale(self, image):
        img = np.array(image)
        interval = 256 / self.color_depth
        for i in range(self.color_depth-1):
            lower_limit = i * interval
            upper_limit = (i + 1) * interval
            img[(img >= lower_limit) & (img < upper_limit)] = lower_limit
        image = Image.fromarray(img.astype('uint8'), 'L')
        return image


# 定义权重映射函数
def map_to_weight(param, lower_bound, upper_bound, intervals):
    interval_size = (upper_bound - lower_bound) / (intervals-1)
    # 将超出上限的值映射为上限，低于下限的值映射为下限
    param.data[param.data >= upper_bound] = upper_bound
    param.data[param.data < lower_bound] = lower_bound
    for i in range(intervals):
        lower_limit = lower_bound + i * interval_size
        upper_limit = lower_bound + (i + 1) * interval_size
        param.data[(param.data >= lower_limit) & (param.data < upper_limit)] = lower_limit


# 定义权重裁剪函数
def clip_weights_generic(model, lower_bound, upper_bound, intervals):
    for param in model.parameters():
        map_to_weight(param, lower_bound, upper_bound, intervals)




