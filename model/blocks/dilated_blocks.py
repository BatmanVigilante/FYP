import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiDilatedBlock(nn.Module):
    """
    Captures multi-scale context using different dilation rates
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.d1 = DilatedConvBlock(in_ch, out_ch, dilation=1)
        self.d2 = DilatedConvBlock(in_ch, out_ch, dilation=2)
        self.d3 = DilatedConvBlock(in_ch, out_ch, dilation=4)

        self.fuse = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        return self.fuse(x)