import torch
import kornia
import torch.fft
import torch.nn as nn
from utils.tools import FilterModule, DCT_mat

class FD(nn.Module):
    def __init__(self, size = 320, kernel=3, in_dim=9):
        super(FD, self).__init__()
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(size)).float(), requires_grad=False
        )
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1),
            requires_grad=False,
        )
        low_filter = FilterModule(size, 0, size // 2.82)
        middle_filter = FilterModule(size, size // 2.82, size // 2)
        high_filter = FilterModule(size, size // 2, size)

        self.filters = nn.ModuleList(
            [high_filter, middle_filter, low_filter]
        )

        self.kernel = kernel
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=in_dim)

    def forward(self, x):
        """频域分解"""
        x_freq = self._DCT_all @ x @ self._DCT_all_T 
        y_list = []
        for i in range(3):
            x_pass = self.filters[i](x_freq) 
            y = self._DCT_all_T @ x_pass @ self._DCT_all  
            y_list.append(y)
        x = torch.cat(y_list, dim=1)

        """噪声信息增强"""
        x_main = kornia.filters.median_blur(x, kernel_size=(self.kernel, self.kernel))
        x_noisy = x - x_main
        x_noisy = self.sigmoid(x_noisy)
        x = x + self.conv(x_noisy)
        return x
