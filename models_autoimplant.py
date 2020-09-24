import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from matplotlib import pyplot as plt


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # print(y.shape)
        # print(x.shape)
        return x * y.expand_as(x)

class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape)
        # print(x.shape)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReplicationPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        # nn.BatchNorm3d(in_features),
                        nn.InstanceNorm3d(in_features),
                        # nn.BatchNorm3d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReplicationPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        # nn.BatchNorm3d(in_features)
                        nn.InstanceNorm3d(in_features)  ]
                        # nn.BatchNorm3d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResidualBlock2d(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock2d, self).__init__()

        conv_block2d = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block2d)

    def forward(self, x):
        return x + self.conv_block(x)

class ReduceBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ReduceBlock, self).__init__()

        conv_block = [  nn.Conv3d(in_features, out_features, 1),
                        nn.Conv3d(out_features, out_features, 3, padding=1),
                        nn.BatchNorm3d(out_features),
                        # nn.BatchNorm3d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReplicationPad3d(1),
                        nn.Conv3d(out_features, out_features, 3),
                        nn.BatchNorm3d(out_features)  ]
                        # nn.BatchNorm3d(out_features)  ]

        conv_block2d = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # print(self.conv_block(x).shape)
        return self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=4):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReplicationPad3d(3),
                    nn.Conv3d(input_nc, 64, 7),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*1
        for _ in range(2):
            model += [  nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                        nn.BatchNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*1

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//1
        for _ in range(2):
            model += [  nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            # model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//1
        # model += [  nn.Upsample(scale_factor=2, mode='nearest'), 
        #             nn.BatchNorm3d(out_features),
        #             nn.ReLU(inplace=True)]

        # model += [  nn.Upsample(scale_factor=2, mode='nearest'), 
        #             # nn.BatchNorm3d(out_features),
        #             nn.ReLU(inplace=True)]
        # for _ in range(2):
        #     model += [  nn.Upsample(scale_factor=2, mode='nearest'),
        #                 nn.BatchNorm3d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features//1

        # Output layer
        model += [nn.Conv3d(64, output_nc, 3, padding=1),
                        nn.Sigmoid() ]
        # model += [  nn.ReplicationPad3d(3),
        #             nn.Conv3d(64, output_nc, 7) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print(slef.model(x).shape)
        return self.model(x)


class Decoder2D(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=4, nf=1):
        super(Decoder2D, self).__init__()

        # Initial convolution block       
        model = [   nn.ReplicationPad2d(3),
                    nn.Conv2d(input_nc, 32, 7),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True) ]

        # # Downsampling
        in_features = 32
        out_features = 32
        # for _ in range(2):
        #     model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #                 nn.BatchNorm2d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features*1

        # Residual blocks
        # model = []
        # # model += [ReduceBlock(nf,32*2)]
        for _ in range(n_residual_blocks):
            # model += [SelfAttention(nf)]
            model += [SELayer2d(32)]
            model += [ResidualBlock2d(32)]

        # Upsampling
        out_features = 32
        for _ in range(2):
            model += [  nn.Upsample(scale_factor=np.sqrt(512/180), mode='nearest'),
            # model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(out_features),
                        # nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//1

        # # upsampling with conv layers
        # out_features = 32
        # for _ in range(2):
        #     # model += [  nn.Upsample(scale_factor=np.sqrt(512/180), mode='nearest'),
        #     model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=1, padding=3, output_padding=0),
        #                 nn.BatchNorm2d(out_features),
        #                 # nn.InstanceNorm3d(out_features),
        #                 nn.ReLU(inplace=True) ]

        # for _ in range(1):
        #     # model += [  nn.Upsample(scale_factor=np.sqrt(512/180), mode='nearest'),
        #     model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=3, padding=2, output_padding=0),
        #                 nn.BatchNorm2d(out_features),
        #                 # nn.InstanceNorm3d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features//1


        # Output layer
        model += [  #nn.ReplicationPad3d(3),
                    nn.Conv2d(32, output_nc, 3, padding=1),
                    nn.ReLU(inplace=True) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('input shape: ', x.shape)
        output = self.model(x)
        # print('output shape: ', output.shape)
        return output


