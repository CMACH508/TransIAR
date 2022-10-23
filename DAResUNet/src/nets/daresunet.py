from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resunet import CBR, BasicBlock, DownSample, DANetHead

class DAResUNet(nn.Module):

    def __init__(self, segClasses=2, k=16, input_channel=1, psp=False):

        super(DAResUNet, self).__init__()

        self.layer0 = CBR(input_channel, k, 7, 1)

        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2 * k),
            BasicBlock(2 * k, 2 * k)
        )

        self.pool2 = DownSample(2 * k, 2 * k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2 * k, 4 * k),
            BasicBlock(4 * k, 4 * k)
        )
        self.pool3 = DownSample(4 * k, 4 * k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4 * k, 8 * k, dilation=1),
            BasicBlock(8 * k, 8 * k, dilation=2),
            BasicBlock(8 * k, 8 * k, dilation=4)
        )

        self.class3 = DANetHead(8 * k, 8 * k)

        # classifier
        self.mlp = torch.nn.Sequential(
            # nn.LayerNorm(13),
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6*6, 1024),  # 27648->1024
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),  # 1024->512
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

        self.class2 = nn.Sequential(
            BasicBlock(4 * k + 8 * k, 8 * k),
            CBR(8 * k, 4 * k, 1)
        )
        self.class1 = nn.Sequential(
            BasicBlock(2 * k + 4 * k, 4 * k),
            CBR(4 * k, 2 * k, 1)
        )
        self.class0 = nn.Sequential(
            BasicBlock(k + 2 * k, 2 * k),
            nn.Conv3d(2 * k, input_channel, kernel_size=1, bias=False)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, is_unet):

        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)

        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)

        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)

        output = self.class3(output3)

        mid = self.mlp(output.view(output.shape[0],-1))
        if not is_unet:
            return {'mid': mid}

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class2(torch.cat([output2, output], 1))
        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class1(torch.cat([output1, output], 1))
        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class0(torch.cat([output0, output], 1))

        return {'y': output, 'mid': mid}


if __name__ == '__main__':
    # aa = torch.rand(size=(1, 1, 80, 80, 80))
    aa = torch.rand(size=(1, 1, 48, 48, 48))
    model = DAResUNet(k=16)
    out = model(aa, is_unet=True)
    print('out.shape=', out['mid'].shape)  # [1, 128, 6, 6, 6] - [1, 2]
    print('out.shape=', out['y'].shape)  # [1, 1, 48, 48, 48]
