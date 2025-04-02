"""
Text2Mel Network implementation based on:
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
https://arxiv.org/abs/1710.08969

Author: Erdene-Ochir Tuguldur
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hparams import HParams as hp
from .layers import E, C, HighwayBlock, GatedConvBlock, ResidualBlock


def Conv(in_channels: int, out_channels: int, kernel_size: int, dilation: int,
         causal: bool = False, nonlinearity: str = 'linear') -> nn.Module:
    return C(in_channels, out_channels, kernel_size, dilation, causal=causal,
             weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization, nonlinearity=nonlinearity)


def BasicBlock(d: int, k: int, delta: int, causal: bool = False) -> nn.Module:
    if hp.text2mel_basic_block == 'gated_conv':
        return GatedConvBlock(d, k, delta, causal=causal,
                              weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization)
    elif hp.text2mel_basic_block == 'highway':
        return HighwayBlock(d, k, delta, causal=causal,
                            weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization)
    else:
        return ResidualBlock(d, k, delta, causal=causal,
                             weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization,
                             widening_factor=2)


def CausalConv(in_channels: int, out_channels: int, kernel_size: int, dilation: int,
               nonlinearity: str = 'linear') -> nn.Module:
    return Conv(in_channels, out_channels, kernel_size, dilation, causal=True, nonlinearity=nonlinearity)


def CausalBasicBlock(d: int, k: int, delta: int) -> nn.Module:
    return BasicBlock(d, k, delta, causal=True)


class TextEnc(nn.Module):
    def __init__(self, vocab: str, e: int = hp.e, d: int = hp.d):
        """
        Text encoder network.
        
        :param vocab: Vocabulary string.
        :param e: Embedding dimension.
        :param d: Text2Mel hidden dimension.
        Input:
            L: (B, N) text inputs.
        Outputs:
            K: (B, d, N) keys.
            V: (B, d, N) values.
        """
        super().__init__()
        self.d = d
        self.embedding = E(len(vocab), e)
        self.layers = nn.Sequential(
            Conv(e, 2 * d, 1, 1, nonlinearity='relu'),
            Conv(2 * d, 2 * d, 1, 1),

            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 3),
            BasicBlock(2 * d, 3, 9), BasicBlock(2 * d, 3, 27),
            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 3),
            BasicBlock(2 * d, 3, 9), BasicBlock(2 * d, 3, 27),

            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 1),

            BasicBlock(2 * d, 1, 1), BasicBlock(2 * d, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        out = self.embedding(x)            # (B, N, e)
        out = out.permute(0, 2, 1)           # (B, e, N)
        out = self.layers(out)               # (B, 2*d, N)
        K, V = out[:, :self.d, :], out[:, self.d:, :]
        return K, V


class AudioEnc(nn.Module):
    def __init__(self, d: int = hp.d, f: int = hp.n_mels):
        """
        Audio encoder network.
        
        :param d: Text2Mel hidden dimension.
        :param f: Number of mel bins.
        Input:
            S: (B, f, T) melspectrograms.
        Output:
            Q: (B, d, T) queries.
        """
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv(f, d, 1, 1, nonlinearity='relu'),
            CausalConv(d, d, 1, 1, nonlinearity='relu'),
            CausalConv(d, d, 1, 1),

            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 3),
            CausalBasicBlock(d, 3, 9), CausalBasicBlock(d, 3, 27),
            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 3),
            CausalBasicBlock(d, 3, 9), CausalBasicBlock(d, 3, 27),

            CausalBasicBlock(d, 3, 3), CausalBasicBlock(d, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AudioDec(nn.Module):
    def __init__(self, d: int = hp.d, f: int = hp.n_mels):
        """
        Audio decoder network.
        
        :param d: Text2Mel hidden dimension.
        :param f: Number of mel bins.
        Input:
            R_prime: (B, 2d, T) concatenation of attended V and Q.
        Output:
            Y: (B, f, T) melspectrogram prediction.
        """
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv(2 * d, d, 1, 1),
            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 3),
            CausalBasicBlock(d, 3, 9), CausalBasicBlock(d, 3, 27),
            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 1),
            CausalBasicBlock(d, 1, 1),
            CausalConv(d, d, 1, 1, nonlinearity='relu'),
            CausalConv(d, f, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Text2Mel(nn.Module):
    def __init__(self, vocab: str, d: int = hp.d):
        """
        Text2Mel network.
        
        :param vocab: Vocabulary string.
        :param d: Text2Mel hidden dimension.
        Inputs:
            L: (B, N) text inputs.
            S: (B, f, T) melspectrograms.
        Outputs:
            Y_logit: Logits for melspectrogram.
            Y: Predicted melspectrogram (after sigmoid).
            A: (B, N, T) attention matrix.
        """
        super().__init__()
        self.d = d
        self.text_enc = TextEnc(vocab, d=d)
        self.audio_enc = AudioEnc()
        self.audio_dec = AudioDec()

    def forward(self, L: torch.Tensor, S: torch.Tensor, monotonic_attention: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        K, V = self.text_enc(L)        # (B, d, N) each
        Q = self.audio_enc(S)           # (B, d, T)
        # Compute scaled dot-product attention scores
        A = torch.bmm(K.transpose(1, 2), Q) / math.sqrt(self.d)  # (B, N, T)

        if monotonic_attention:
            # Optionally enforce a roughly monotonic attention alignment.
            # This is a simple (non-vectorized) implementation; further optimization is recommended.
            B, N, T = A.size()
            for i in range(B):
                prev_index = 0
                for t in range(T):
                    # Find best matching text position at time step t
                    _, current_index = A[i, :, t].max(dim=0)
                    if abs(current_index.item() - prev_index) > 3:
                        # Penalize positions far from the previous best index
                        A[i, :, t] = -1e9
                        # Force a slight forward shift
                        forced_index = min(N - 1, prev_index + 1)
                        A[i, forced_index, t] = 1.0
                    prev_index = A[i, :, t].argmax().item()

        A = F.softmax(A, dim=1)  # Softmax over the text dimension
        R = torch.bmm(V, A)      # (B, d, T)
        R_prime = torch.cat((R, Q), dim=1)  # (B, 2*d, T)
        Y_logit = self.audio_dec(R_prime)
        Y = torch.sigmoid(Y_logit)
        return Y_logit, Y, A
