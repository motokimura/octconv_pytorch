# an unofficial implementation for OctConv. 
# reference:
# https://github.com/iacolippo/octconv-pytorch/blob/master/octconv.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OctConv2d(nn.Module):
    """OctConv proposed in:
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.
    paper link: https://arxiv.org/abs/1904.05049
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, alphas=(0.5, 0.5)):
        super(OctConv2d, self).__init__()

        alpha_in, alpha_out = alphas
        assert (0 <= alpha_in <= 1) and (0 <= alpha_in <= 1), "Alphas must be in interval [0, 1]"

        # input channels
        self.ch_in_lf = int(alpha_in * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        # output channels
        self.ch_out_lf = int(alpha_out * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf

        # padding: (H - F + 2P)/S + 1 = 2 * [(0.5 H - F + 2P)/S +1] -> P = (F-S)/2
        padding = (kernel_size - stride) // 2

        # conv layers
        self.HtoH, self.HtoL, self.LtoH, self.LtoL = None, None, None, None
        if not (self.ch_out_hf == 0 or self.ch_in_hf == 0):
            self.HtoH = nn.Conv2d(self.ch_in_hf, self.ch_out_hf, kernel_size, stride, padding, bias=bias)
        if not (self.ch_out_lf == 0 or self.ch_in_hf == 0):
            self.HtoL = nn.Conv2d(self.ch_in_hf, self.ch_out_lf, kernel_size, stride, padding, bias=bias)
        if not (self.ch_out_hf == 0 or self.ch_in_lf == 0):
            self.LtoH = nn.Conv2d(self.ch_in_lf, self.ch_out_hf, kernel_size, stride, padding, bias=bias)
        if not (self.ch_out_lf == 0 or self.ch_in_lf == 0):
            self.LtoL = nn.Conv2d(self.ch_in_lf, self.ch_out_lf, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        hf, lf = None, None
        # logic to handle input tensors:
        # if ch_in_lf = 0., we assume to be at the first layer, with only high freq repr
        if self.ch_in_lf == 0:
            hf = x
        elif self.ch_in_hf == 0:
            lf = x
        else:
            hf, lf = x

        # apply convolutions
        oHtoH = oHtoL = oLtoH = oLtoL = 0.
        if self.HtoH is not None:
            oHtoH = self.HtoH(hf)
        if self.HtoL is not None:
            oHtoL = self.HtoL(F.avg_pool2d(hf, 2))
        if self.LtoH is not None:
            oLtoH = F.interpolate(self.LtoH(lf), scale_factor=2, mode='nearest')
        if self.LtoL is not None:
            oLtoL = self.LtoL(lf)

        # compute output tensors
        hf = oHtoH + oLtoH
        lf = oLtoL + oHtoL

        # logic to handle output tensors:
        # if ch_out_lf = 0., we assume to be at the last layer, with only high freq repr
        if self.ch_out_lf == 0:
            return hf
        elif self.ch_out_hf == 0:
            return lf
        else:
            # if alpha in (0, 1)
            return (hf, lf)


class OctConvMaxPool2d(nn.Module):
    """Pooling module for 2d features represented by OctConv way.
    """
    def __init__(self, channels, kernel_size, stride=None, alpha=0.5):
        super(OctConvMaxPool2d, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.ch_hf == 0 or self.ch_lf == 0:
            return F.max_pool2d(x, self.kernel_size, self.stride)

        hf, lf = x
        hf = F.max_pool2d(hf, self.kernel_size, self.stride)
        lf = F.max_pool2d(lf, self.kernel_size, self.stride)
        return (hf, lf)


class OctConvUpsample(nn.Module):
    def __init__(self, channels, scale_factor, mode='bilinear', alpha=0.5):
        super(OctConvUpsample, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.ch_hf == 0 or self.ch_lf == 0:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        hf, lf = x
        hf = F.interpolate(hf, scale_factor=self.scale_factor, mode=self.mode)
        lf = F.interpolate(lf, scale_factor=self.scale_factor, mode=self.mode)
        return (hf, lf)


class OctConvBatchNorm2d(nn.Module):
    def __init__(self, channels, alpha=0.5):
        super(OctConvBatchNorm2d, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        # prepare batchnorm layers for lf and hf features
        self.bn_lf = nn.BatchNorm2d(self.ch_lf) if self.ch_lf > 0 else None
        self.bn_hf = nn.BatchNorm2d(self.ch_hf) if self.ch_hf > 0 else None

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.bn_lf is None:
            return self.bn_hf(x)
        if self.bn_hf is None:
            return self.bn_lf(x)

        hf, lf = x
        hf = self.bn_hf(hf)
        lf = self.bn_lf(lf)
        return (hf, lf)


class OctConvReLU(nn.Module):
    def __init__(self, channels, alpha=0.5):
        super(OctConvReLU, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.ch_hf == 0 or self.ch_lf == 0:
            return F.relu(x, inplace=True)

        hf, lf = x
        hf = F.relu(hf, inplace=True)
        lf = F.relu(lf, inplace=True)
        return (hf, lf)
