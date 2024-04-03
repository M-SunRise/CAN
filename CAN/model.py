import torch
import torch.nn as nn

from CAN.FD import FD
from CAN.AFE import AFE
from CAN.MAF import MAF
from CAN.CAB import CAB
from backbone.EfficientNetB4 import EfficientNetB4, switch_layers


class CAN(nn.Module):
    def __init__(self, num_classes=1):
        super(CAN, self).__init__()
        # self.backbone = EfficientNetB4(pretrained = True, num_classes = num_classes, act_layer=nn.Mish)
        self.backbone = EfficientNetB4(pretrained = True, num_classes = num_classes)
        self.freq_backbone = EfficientNetB4(pretrained = True, num_classes = num_classes, act_layer=nn.Mish)

        self.fd = FD(size = 320, kernel=3, in_dim=9)
        self.afe = AFE(patch_size=[2,4,8], hidden_dim=3, out_dim=48)
        self.maf = MAF(in_dim = 32, hidden_dim = 32, out_dim = 32)

        self.cabList0 = nn.Sequential(
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1),
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1),
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1),
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1),
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1),
                            CAB(dim=32, num_heads=1, center_size=2, around_size=2, alpha=0.5, down_ratio=1)
                        )
        self.cabList1 = nn.Sequential(
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                            CAB(dim=56, num_heads=1, center_size=2, around_size=4, alpha=0.5, down_ratio=2),
                        )

    def forward(self, x):
        freq = self.afe(self.fd(x))
        x = switch_layers(self.backbone, 'stem', x)
        x = switch_layers(self.backbone, 'blocks', x, 0, 1)
        freq = switch_layers(self.freq_backbone, 'blocks', freq, 0, 1)
        x = self.maf(x, freq)
        x = self.cabList0(x)
        x = switch_layers(self.backbone, 'blocks', x, 2, 2)
        x = self.cabList1(x)
        x = switch_layers(self.backbone, 'blocks', x, 3, 6)
        x = switch_layers(self.backbone, 'head', x)
        fea, x = switch_layers(self.backbone, 'fc', x)
        result = {"logits": x, "features": fea}
        return result
    
if __name__ == '__main__': 
    model = CAN(num_classes=1)
    dummy = torch.rand((4,3,320,320))

    
