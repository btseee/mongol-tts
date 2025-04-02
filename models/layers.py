"""
Neural network layers for the TTS models.

Author: Erdene-Ochir Tuguldur

Exports:
    E, D, C, HighwayBlock, GatedConvBlock, ResidualBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hparams import HParams as hp


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        """Custom LayerNorm that permutes inputs for 1D convolution."""
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, C, T) -> permute to (B, T, C)
        x = x.permute(0, 2, 1)
        y = super().forward(x)
        return y.permute(0, 2, 1)


class D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int,
                 weight_init: str = 'none', normalization: str = 'weight', nonlinearity: str = 'linear'):
        """1D Deconvolution layer."""
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                         stride=2,  # always use stride=2 for deconvolution (per paper)
                                         dilation=dilation)

        if normalization == 'weight':
            self.deconv = nn.utils.weight_norm(self.deconv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.deconv.weight, nn.init.calculate_gain(nonlinearity))

    def forward(self, x: torch.Tensor, output_size: int = None) -> torch.Tensor:
        y = self.deconv(x, output_size=output_size)
        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


class C(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int,
                 causal: bool = False, weight_init: str = 'none', normalization: str = 'weight',
                 nonlinearity: str = 'linear'):
        """
        1D Convolution layer.
        
        :param causal: When True, uses causal padding.
        """
        super().__init__()
        self.causal = causal
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=self.padding, dilation=dilation)

        if normalization == 'weight':
            self.conv = nn.utils.weight_norm(self.conv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.causal and self.padding > 0:
            # Remove extra timesteps at the end for causal convolutions
            y = y[:, :, :-self.padding]
        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


class E(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Embedding layer."""
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class HighwayBlock(nn.Module):
    def __init__(self, d: int, k: int, delta: int, causal: bool = False,
                 weight_init: str = 'none', normalization: str = 'weight'):
        """
        Highway network layer as in https://arxiv.org/abs/1505.00387.
        Input and output shapes remain the same.
        """
        super().__init__()
        self.d = d
        self.conv = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta,
                      causal=causal, weight_init=weight_init, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.conv(x)
        H1, H2 = L[:, :self.d, :], L[:, self.d:, :]
        return torch.sigmoid(H1) * H2 + (1 - torch.sigmoid(H1)) * x


class GatedConvBlock(nn.Module):
    def __init__(self, d: int, k: int, delta: int, causal: bool = False,
                 weight_init: str = 'none', normalization: str = 'weight'):
        """
        Gated convolution block as in https://arxiv.org/abs/1612.08083.
        """
        super().__init__()
        self.conv = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta,
                      causal=causal, weight_init=weight_init, normalization=normalization)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.conv(x)
        return self.glu(L) + x


class ResidualBlock(nn.Module):
    def __init__(self, d: int, k: int, delta: int, causal: bool = False,
                 weight_init: str = 'none', normalization: str = 'weight', widening_factor: int = 2):
        """
        Residual block as in https://arxiv.org/abs/1512.03385.
        """
        super().__init__()
        self.conv1 = C(in_channels=d, out_channels=widening_factor * d, kernel_size=k, dilation=delta,
                       causal=causal, weight_init=weight_init, normalization=normalization, nonlinearity='relu')
        self.conv2 = C(in_channels=widening_factor * d, out_channels=d, kernel_size=k, dilation=delta,
                       causal=causal, weight_init=weight_init, normalization=normalization, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x
