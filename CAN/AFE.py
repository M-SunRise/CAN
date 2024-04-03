import torch
import torch.fft
import torch.nn as nn
class AFE(nn.Module):   
    def __init__(self, patch_size=[2,4,8], hidden_dim=3, out_dim=48):
        super(AFE, self).__init__()

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = out_dim // 2 ** i
            else:
                dim = out_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(hidden_dim, dim, kernel_size=ps, stride=stride, padding=padding))
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.Mish(inplace=True)
        
    def forward(self, x):

        x = torch.split(x, 3, 1)
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x[i])
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=1)
        x = self.act(self.norm(x))
        return x