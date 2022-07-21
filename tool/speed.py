import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import os
import time
import math
import torch
import torch.nn as nn
from typing import Any, Callable


#########################   BPNet   #########################
'''  weight init  '''
def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(m, "weight"):
            if init_type == "gaussian":
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


# #########################   Generator   ###############################
class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=None):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.conv.apply(weights_init("gaussian"))
        self.after: Any[Callable]
        self.before: Any[Callable]
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = nn.Tanh()
        elif after == "sigmoid":
            self.after = nn.Sigmoid()

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        print()

    def forward(self, x):
        if hasattr(self, "before"):
            x = self.before(x)
        x = self.conv(x)
        if hasattr(self, "after"):
            x = self.after(x)
        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=None):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv.apply(weights_init("gaussian"))
        if after == "BN":
            self.after = nn.BatchNorm2d(out_channels)
        elif after == "Tanh":
            self.after = nn.Tanh()
        elif after == "sigmoid":
            self.after = nn.Sigmoid()

        if before == "ReLU":
            self.before = nn.ReLU(inplace=True)
        elif before == "LReLU":
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "before"):
            x = self.before(x)
        x = self.conv(x)
        if hasattr(self, "after"):
            x = self.after(x)
        return x


class Generator_nobgc(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator_nobgc, self).__init__()

        self.Cv0   = Cvi (in_channels, 64)
        self.Cv1   = Cvi (64,   128,          before="LReLU", after="BN")
        self.Cv2   = Cvi (128,  256,          before="LReLU", after="BN")
        self.Cv3   = Cvi (256,  512,          before="LReLU", after="BN")
        self.Cv4   = Cvi (512,  512,          before="LReLU", after="BN")
        self.Cv5   = Cvi (512,  512,          before="LReLU")
        self.CvT6  = CvTi(512,  512,          before="ReLU",  after="BN")
        self.CvT7  = CvTi(1024, 512,          before="ReLU",  after="BN")
        self.CvT8  = CvTi(1024, 256,          before="ReLU",  after="BN")
        self.CvT9  = CvTi(512,  128,          before="ReLU",  after="BN")
        self.CvT10 = CvTi(256,  64,           before="ReLU",  after="BN")
        self.CvT11 = CvTi(128,  out_channels, before="ReLU",  after="Tanh")

    def forward(self, img):
        # encoder
        input = img
        x0   = self.Cv0(input)
        x1   = self.Cv1(x0)
        x2   = self.Cv2(x1)
        x3   = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5   = self.Cv5(x4_3)

        # decoder
        x6     = self.CvT6(x5)
        cat1_1 = torch.cat([x6, x4_3], dim=1)
        x7_1   = self.CvT7(cat1_1)
        cat1_2 = torch.cat([x7_1, x4_2], dim=1)
        x7_2   = self.CvT7(cat1_2)
        cat1_3 = torch.cat([x7_2, x4_1], dim=1)
        x7_3   = self.CvT7(cat1_3)

        cat2 = torch.cat([x7_3, x3], dim=1)
        x8 = self.CvT8(cat2)

        cat3 = torch.cat([x8, x2], dim=1)
        x9 = self.CvT9(cat3)

        cat4 = torch.cat([x9, x1], dim=1)
        x10 = self.CvT10(cat4)

        cat5 = torch.cat([x10, x0], dim=1)
        out = self.CvT11(cat5)

        return out


def main():
    model = Generator_nobgc().cuda()
    print("BPNet parameters: ", sum(param.numel() for param in model.parameters())/1e6)
    model.eval()
    with torch.no_grad():
        # image = torch.randn(1, 3, 4096, 2304).cuda() # 2160
        image = torch.randn(1, 3, 2048, 1280).cuda() # 1920 1080
        for i in range(50):
            model(image)
        print("throughput averaged with 100 times")
        tic1 = time.time()
        for i in range(100):
            model(image)
        tic2 = time.time()
        print((tic2-tic1)/100)


if __name__ == '__main__':
    main()