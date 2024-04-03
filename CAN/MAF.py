import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class MAF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MAF, self).__init__()

        self.conv1 = nn.Conv2d(
            in_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_dim ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_dim, out_dim, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )   #[1, 4096, 32]
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3)) #[1, 32, 4096]

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)    #[1, 4096, 4096]

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )       #[1, 4096, 32]
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()
        #[1, 32, 64, 64]
        output = rgb + self.conv4(z)

        return output