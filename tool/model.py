import math
import torch
import torch.nn as nn
import torchvision.utils as utils
from typing import Any, Callable


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


'''  Light Prediction Network  '''
class LPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(LPNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.color_pred = nn.Sequential(nn.Linear(128, out_channels), nn.Tanh())

    def forward(self, x):
        x = self.features(x)
        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.color_pred(x)
        return x


'''  Discriminator  '''
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator, self).__init__()

        # shallow
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        self.shallow = nn.Sequential(*sequence)
        # increase filters
        conv = [nn.Conv2d(ndf,   ndf*2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf*4),
                nn.LeakyReLU(0.2, True)]
        self.conv = nn.Sequential(*conv)
        # last filters
        conv_last = [nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
                    nn.InstanceNorm2d(ndf*8),
                    nn.LeakyReLU(0.2, True),]
        self.conv_last = nn.Sequential(*conv_last)
        # tail
        self.tail = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)  # output 1 channel prediction map

    def forward(self, input):
        """Standard forward."""
        shallow   = self.shallow(input)
        conv      = self.conv(shallow)
        conv_last = self.conv_last(conv)
        output    = self.tail(conv_last)
        return output


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


class Generator6(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Generator6, self).__init__()
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

    def forward(self, img, bgc):
        # encoder
        input = torch.cat([img, bgc], dim=1)
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


class Generator3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator3, self).__init__()
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


if __name__=="__main__":

    input = torch.ones((2, 3, 256, 256))
    bgc   = torch.ones((2, 3, 256, 256))

    # model = NLayerDiscriminator()
    model = Generator6()
    output = model(input, bgc)
    print(output.size())
    print("netG parameters: ", sum(param.numel() for param in model.parameters())/1e6)
