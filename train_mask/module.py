import torch
import torch.nn as nn
import torch.nn.functional as F

def safeshape(sec, pri):
    sh = pri.shape
    sec = sec[..., :sh[-2], :sh[-1]]
    return sec

class Identity(nn.Module):
    def forward(self, x):
        return x

class UNet2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ks, pd):
        super(UNet2dBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.ks = ks
        self.pd = pd
        
        activation_fn = nn.ReLU(inplace=True)

        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=1, padding=pd),
            activation_fn,
            nn.GroupNorm(1, out_c),
            nn.Conv2d(out_c, out_c, kernel_size=ks, stride=1, padding=pd),
            activation_fn,
            nn.GroupNorm(1, out_c)
        )

    def forward(self, x):
        return self.model(x)

class MPINet2d(nn.Module):
    def __init__(self, in_c, out_c, num_depths, nf=64):
        super(MPINet2d, self).__init__()

        self.num_depths = num_depths

        self.block0 = UNet2dBlock(in_c, nf, 7, 3)
        self.block1 = UNet2dBlock(nf, nf*2, 5, 2)
        self.block2 = UNet2dBlock(nf*2, nf*4, 3, 1)
        self.block3 = UNet2dBlock(nf*4, nf*4, 3, 1)

        self.block4 = UNet2dBlock(nf*8, nf*4, 3, 1)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block5 = UNet2dBlock(nf*6, nf*2, 3, 1)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block6 = UNet2dBlock(nf*3, nf, 3, 1)
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = nn.Conv2d(nf, out_c, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, dummy=None):
        conv1 = self.block0(x)
        conv2 = self.block1(F.avg_pool2d(conv1, 2))
        conv3 = self.block2(F.avg_pool2d(conv2, 2))
        conv4 = self.block3(F.avg_pool2d(conv3, 2))

        conv5 = self.block4(torch.cat([self.up4(conv4), conv3], 1))
        conv6 = self.block5(torch.cat([self.up5(conv5), conv2], 1))
        conv7 = self.block6(torch.cat([self.up6(conv6), conv1], 1))
        out = self.out(conv7)

        return (self.tanh(out) + 1.0) / 2.0

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ColorNet2d(nn.Module):
    def __init__(self, in_c, out_c, nf=128):
        super(ColorNet2d, self).__init__()

        self.block0 = Conv2dBlock(in_c, nf, 7, 1, padding=3, norm='in', pad_type='reflect')
        self.block1 = Conv2dBlock(nf, nf//2, 7, 1, padding=3,  norm='in', pad_type='reflect')
        self.block2 = Conv2dBlock(nf//2, nf//4, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.block3 = Conv2dBlock(nf//4, nf//8, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.out = Conv2dBlock(nf//8, out_c, 3, 1, padding=1, pad_type='reflect')

    def forward(self, x):
        conv1 = self.block0(x)
        conv2 = self.block1(conv1)
        conv3 = self.block2(conv2)
        conv4 = self.block3(conv3)

        out = self.out(conv4)
        
        weights = torch.cat([torch.zeros_like(out[:, :1]), out], 1)
        weights = F.softmax(weights, 1)
       
        return weights

class FeatureNet2d(nn.Module):
    def __init__(self, in_c, out_c, nf=16):
        super(FeatureNet2d, self).__init__()

        self.block0 = Conv2dBlock(in_c, nf, 7, 2, padding=3, norm='in', pad_type='reflect')
        self.block1 = Conv2dBlock(nf, nf, 3, 2, padding=1,  norm='in', pad_type='reflect')
        self.block2 = Conv2dBlock(nf, nf, 3, 2, padding=1, norm='in', pad_type='reflect')

        self.block3 = Conv2dBlock(nf*2, nf, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block4 = Conv2dBlock(nf*2, nf//2, 3, 1, padding=1,  norm='in', pad_type='reflect')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block5 = Conv2dBlock(nf//2+in_c, nf//2, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out = Conv2dBlock(nf//2, out_c, 3, 1, padding=1, activation='tanh', pad_type='reflect')

    def forward(self, x):

        conv1 = self.block0(x)
        conv2 = self.block1(conv1)
        conv3 = self.block2(conv2)

        conv4 = self.block3(torch.cat([self.up3(conv3), conv2], 1))
        conv5 = self.block4(torch.cat([self.up4(conv4), conv1], 1))
        conv6 = self.block5(torch.cat([self.up5(conv5), x], 1))
        out = self.out(conv6)

        return out

class AppearanceNet2d(nn.Module):
    def __init__(self, in_c, out_c, nf=16):
        super(AppearanceNet2d, self).__init__()

        self.block0 = Conv2dBlock(in_c, nf, 7, 2, padding=3, norm='in', pad_type='reflect')
        self.block1 = Conv2dBlock(nf, nf, 3, 2, padding=1,  norm='in', pad_type='reflect')
        self.block2 = Conv2dBlock(nf, nf, 3, 2, padding=1, norm='in', pad_type='reflect')

        self.block3 = Conv2dBlock(nf*2, nf, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block4 = Conv2dBlock(nf*2, nf//2, 3, 1, padding=1,  norm='in', pad_type='reflect')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block5 = Conv2dBlock(nf//2+in_c, nf//2, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out = Conv2dBlock(nf//2, out_c, 3, 1, padding=1, activation='sigmoid', pad_type='reflect')

    def forward(self, x):

        conv1 = self.block0(x)
        conv2 = self.block1(conv1)
        conv3 = self.block2(conv2)

        conv4 = self.block3(torch.cat([self.up3(conv3), conv2], 1))
        conv5 = self.block4(torch.cat([self.up4(conv4), conv1], 1))
        conv6 = self.block5(torch.cat([self.up5(conv5), x], 1))
        out = self.out(conv6)

        return out
class Conv3dBlock(nn.Module):
    def __init__(self, in_c, out_c, ks, pd, s=1,
            activation='relu', norm_layer='layer_norm'):
        super(Conv3dBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.ks = ks
        self.pd = pd
        self.s = s
        
        if activation == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'identity':
            activation_fn = Identity()

        if norm_layer == 'layer_norm':
            norm_fn = nn.GroupNorm(1, out_c)
        elif norm_layer == 'identity':
            norm_fn = Identity()

        self.model = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=ks, stride=s, padding=pd),
            activation_fn,
            norm_fn
        )

    def forward(self, x):
        return self.model(x)

class MPINet3d(nn.Module):
    def __init__(self, in_c, out_c, nf=8):
        super(MPINet3d, self).__init__()

        self.conv1_1 = Conv3dBlock(in_c, nf, 7, 3, 1)
        self.conv1_2 = Conv3dBlock(nf, nf*2, 7, 3, 2)

        self.conv2_1 = Conv3dBlock(nf*2, nf*2, 3, 1, 1)
        self.conv2_2 = Conv3dBlock(nf*2, nf*4, 3, 1, 2)

        self.conv3_1 = Conv3dBlock(nf*4, nf*4, 3, 1, 1)
        self.conv3_2 = Conv3dBlock(nf*4, nf*8, 3, 1, 2)

        self.conv4_1 = Conv3dBlock(nf*8, nf*8, 3, 1, 1)
        self.conv4_2 = Conv3dBlock(nf*8, nf*8, 3, 1, 1)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5_1 = Conv3dBlock(nf*16, nf*4, 3, 1, 1)
        self.conv5_2 = Conv3dBlock(nf*4, nf*4, 3, 1, 1)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_1 = Conv3dBlock(nf*8, nf*2, 3, 1, 1)
        self.conv6_2 = Conv3dBlock(nf*2, nf*2, 3, 1, 1)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_1 = Conv3dBlock(nf*4, nf*2, 3, 1, 1)
        self.conv7_2 = Conv3dBlock(nf*2, nf, 3, 1, 1)
        self.conv7_3 = Conv3dBlock(nf, out_c, 3, 1, 1, activation='identity', norm_layer='identity')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        conv1 = self.conv1_2(self.conv1_1(x))
        conv2 = self.conv2_2(self.conv2_1(conv1))
        conv3 = self.conv3_2(self.conv3_1(conv2))
        conv4 = self.conv4_2(self.conv4_1(conv3))

        x = torch.cat([conv4, conv3], 1)
        conv5 = self.conv5_2(self.conv5_1(self.up5(x)))
        x = torch.cat([safeshape(conv5, conv2), conv2], 1)
        conv6 = self.conv6_2(self.conv6_1(self.up6(x)))
        x = torch.cat([safeshape(conv6, conv1), conv1], 1)
        x = self.conv7_3(self.conv7_2(self.conv7_1(self.up7(x))))

        weights = x[:, 1:] # blending weights between views
        weights = torch.cat([torch.zeros_like(x[:, :1]), weights], 1)
        weights = F.softmax(weights, 1)
        alpha = self.sigmoid(x[:, :1])

        return torch.cat([alpha, weights], 1)

class MaskNet3d(nn.Module):
    def __init__(self, in_c, out_c, nf=8):
        super(MaskNet3d, self).__init__()

        self.conv1_1 = Conv3dBlock(in_c, nf, 7, 3, 1)
        self.conv1_2 = Conv3dBlock(nf, nf*2, 7, 3, 2)

        self.conv2_1 = Conv3dBlock(nf*2, nf*2, 3, 1, 1)
        self.conv2_2 = Conv3dBlock(nf*2, nf*4, 3, 1, 2)

        self.conv3_1 = Conv3dBlock(nf*4, nf*4, 3, 1, 1)
        self.conv3_2 = Conv3dBlock(nf*4, nf*8, 3, 1, 2)

        self.conv4_1 = Conv3dBlock(nf*8, nf*8, 3, 1, 1)
        self.conv4_2 = Conv3dBlock(nf*8, nf*8, 3, 1, 1)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5_1 = Conv3dBlock(nf*16, nf*4, 3, 1, 1)
        self.conv5_2 = Conv3dBlock(nf*4, nf*4, 3, 1, 1)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_1 = Conv3dBlock(nf*8, nf*2, 3, 1, 1)
        self.conv6_2 = Conv3dBlock(nf*2, nf*2, 3, 1, 1)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_1 = Conv3dBlock(nf*4, nf*2, 3, 1, 1)
        self.conv7_2 = Conv3dBlock(nf*2, nf, 3, 1, 1)
        self.conv7_3 = Conv3dBlock(nf, out_c, 3, 1, 1, activation='identity', norm_layer='identity')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        conv1 = self.conv1_2(self.conv1_1(x))
        conv2 = self.conv2_2(self.conv2_1(conv1))
        conv3 = self.conv3_2(self.conv3_1(conv2))
        conv4 = self.conv4_2(self.conv4_1(conv3))

        x = torch.cat([conv4, conv3], 1)
        conv5 = self.conv5_2(self.conv5_1(self.up5(x)))
        x = torch.cat([safeshape(conv5, conv2), conv2], 1)
        conv6 = self.conv6_2(self.conv6_1(self.up6(x)))
        x = torch.cat([safeshape(conv6, conv1), conv1], 1)
        x = self.conv7_3(self.conv7_2(self.conv7_1(self.up7(x))))

        return self.sigmoid(x)