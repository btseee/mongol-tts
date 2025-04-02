"""
SSRN Network implementation based on:
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
https://arxiv.org/abs/1710.08969

Author: Erdene-Ochir Tuguldur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.hparams import HParams as hp
from .layers import D, C, HighwayBlock, GatedConvBlock, ResidualBlock


def Conv(in_channels: int, out_channels: int, kernel_size: int, dilation: int,
         nonlinearity: str = 'linear') -> nn.Module:
    return C(in_channels, out_channels, kernel_size, dilation, causal=False,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def DeConv(in_channels: int, out_channels: int, kernel_size: int, dilation: int,
           nonlinearity: str = 'linear') -> nn.Module:
    return D(in_channels, out_channels, kernel_size, dilation,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def BasicBlock(d: int, k: int, delta: int) -> nn.Module:
    if hp.ssrn_basic_block == 'gated_conv':
        return GatedConvBlock(d, k, delta, causal=False,
                              weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    elif hp.ssrn_basic_block == 'highway':
        return HighwayBlock(d, k, delta, causal=False,
                            weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    else:
        return ResidualBlock(d, k, delta, causal=False,
                             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization,
                             widening_factor=1)


class SSRN(nn.Module):
    def __init__(self, c: int = hp.c, f: int = hp.n_mels, f_prime: int = (1 + hp.n_fft // 2)):
        """
        Spectrogram super-resolution network.
        
        :param c: SSRN dimensionality.
        :param f: Number of mel bins.
        :param f_prime: Full spectrogram dimensionality.
        Input:
            Y: (B, f, T) predicted melspectrograms.
        Outputs:
            Z_logit: logits for full spectrogram.
            Z: (B, f_prime, 4*T) full spectrograms (after sigmoid).
        """
        super().__init__()
        self.layers = nn.Sequential(
            Conv(f, c, 1, 1),

            BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),
            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            Conv(c, 2 * c, 1, 1),

            BasicBlock(2 * c, 3, 1), BasicBlock(2 * c, 3, 1),

            Conv(2 * c, f_prime, 1, 1),

            BasicBlock(f_prime, 1, 1),

            Conv(f_prime, f_prime, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        Z_logit = self.layers(x)
        Z = torch.sigmoid(Z_logit)
        return Z_logit, Z
