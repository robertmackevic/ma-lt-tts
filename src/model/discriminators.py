from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from src.model.commons import get_padding
from src.model.modules import LRELU_SLOPE


class DiscriminatorP(nn.Module):
    def __init__(self,
                 *,
                 period: int,
                 kernel_size: int = 5,
                 stride: int = 3,
                 use_spectral_norm: bool = False):
        super(DiscriminatorP, self).__init__()

        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])

        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape

        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self,
                 *,
                 use_spectral_norm: bool = False):
        super(DiscriminatorS, self).__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self,
                 *,
                 use_spectral_norm: bool = False):
        super(MultiPeriodDiscriminator, self).__init__()

        periods = [2, 3, 5, 7, 11]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(period=i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
