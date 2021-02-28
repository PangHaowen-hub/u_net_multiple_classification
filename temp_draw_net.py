import hiddenlayer as hl
import torch
import torch.nn as nn

def passthrough(x, **kwargs):
    return x

class LUConv(nn.Module):
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.modules.InstanceNorm3d(nchan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.modules.InstanceNorm3d(outChans, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 0)  # 将输入复制为16个通道，便于res
        out = self.relu1(torch.add(out, x16))  # res操作
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)  # 下采样
        self.bn1 = nn.modules.InstanceNorm3d(outChans, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.do1 = passthrough
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.modules.InstanceNorm3d(outChans // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=3, padding=1)
        self.bn1 = nn.modules.InstanceNorm3d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # if nll:
        #     self.softmax = F.log_softmax
        # else:
        #     self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    def __init__(self, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        # self.down_tr128 = DownTransition(64, 3, dropout=True)
        # self.down_tr256 = DownTransition(128, 2, dropout=True)
        # self.up_tr256 = UpTransition(256, 256, 2, dropout=True)
        # self.up_tr128 = UpTransition(256, 128, 2, dropout=True)
        # self.up_tr64 = UpTransition(128, 64, 1)
        # self.up_tr32 = UpTransition(64, 32, 1)
        # self.out_tr = OutputTransition(32, nll)

    ## The network topology as described in the diagram in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    ## the number of convolutions in each layer corresponds to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        # out128 = self.down_tr128(out64)
        # out256 = self.down_tr256(out128)
        # out = self.up_tr256(out256, out128)
        # out = self.up_tr128(out, out64)
        # out = self.up_tr64(out, out32)
        # out = self.up_tr32(out, out16)
        # out = self.out_tr(out)
        # 多分类softmax 二分类sigmod
        return out64




model = VNet()
x = hl.build_graph(model, torch.zeros([1, 1, 100, 256, 256]))
x.save('net.pdf')