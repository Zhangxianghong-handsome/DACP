import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)  
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        kernel_height, kernel_width = self.weight.shape[2], self.weight.shape[3]
        patches = F.unfold(
            x, kernel_size=(kernel_height, kernel_width), stride=self.stride, padding=self.padding
        )
        patches = patches.view(x.size(0), x.size(1), kernel_height, kernel_width, -1)
        patches = patches.permute(0, 4, 1, 2, 3)  
        expanded_weight = self.weight[None, :, None, :, :, :]  

        expanded_patches = patches[:, None, :, :, :, :]   

        weighted_sum = expanded_patches + expanded_weight  

        exp_result = 1 - torch.exp(weighted_sum)

        sum_result = exp_result.sum(dim=(3, 4, 5))  

        H = (x.size(2) + 2 * self.padding - kernel_height) // self.stride + 1  
        W = (x.size(3) + 2 * self.padding- kernel_width) // self.stride + 1  

        out = sum_result.view(x.size(0), self.weight.size(0), H, W)  


        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            CustomConv2d(1, 10, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ForecastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            CustomConv2d(3, 10, 3, 1, 1),
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
