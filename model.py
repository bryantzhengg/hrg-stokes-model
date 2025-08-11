import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFlowNet(nn.Module):
    """
    input : (B, 2, H, W) -> [mask, inlet]
    output: (B, 2, H, W) -> [u, v]
    """
    def __init__(self, in_ch=2, mid=16, out_ch=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.bot = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 3, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bot(z)
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        y = self.dec(z)
        return y
