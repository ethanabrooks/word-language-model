from typing import Optional, Tuple

import torch
from torch.nn import Parameter, Linear
from torch.nn.init import xavier_uniform_

import models
import multihead_attention
import transformer
from torch import Tensor


def with_last_col_1(x: Tensor, last1=None):
    if last1 is None:
        last1 = torch.zeros_like(x)
        last1[..., -1] = 1
    return x - x * last1 + last1


def with_first_col_1(x: Tensor, first1=None):
    if first1 is None:
        first1 = torch.zeros_like(x)
        first1[..., 0] = 1
    return x - x * first1 + first1


def scan(x: Tensor) -> Tensor:
    return with_last_col_1((1 - x).cumprod(-1)).roll(1, -1) * x


def scan_forward(x: Tensor) -> Tensor:
    x = with_last_col_1(x)
    x = torch.cat([row.roll(-i, -1) for i, row in enumerate(x.split(1, -2))], -2)
    x = scan(x)
    return torch.cat([row.roll(i, -1) for i, row in enumerate(x.split(1, -2))], -2)


def scan_backward(x: Tensor) -> Tensor:
    *_, n = x.shape
    x = with_last_col_1(x.flip(-1))
    n -= 1
    x = torch.cat([row.roll(i - n, -1) for i, row in enumerate(x.split(1, -2))], -2)
    x = scan(x)
    x = torch.cat([row.roll(n - i, -1) for i, row in enumerate(x.split(1, -2))], -2)
    return x.flip(-1)


class MultiheadAttention(multihead_attention.MultiheadAttention):
    # noinspection PyPep8Naming
    def get_attn_output_weights(self, k, q):  # type: (Tensor, Tensor) -> Tensor
        r"""
        Args:
            q, k: map a query and a set of keys to an attention mask.
                See "Attention Is All You Need" for more details.


        Shape:
            Inputs:
            - q: :math:`(B * H, L, E)` where B is the batch size, H is the number of heads, L is the target sequence
            length and E is the head dimension.
            - k: :math:`(B * H, S, E)`, where B is the batch size, H is the number of heads, S is the source sequence
            length and E is the head dimension.

            Outputs:
            - attn_output_weights: :math:`(B * H, L, S)` where B is the batch size, H is the number of heads, L is the
            is the target sequence length and S is the source sequence length.
        """
        L = q.size(1)
        S = k.size(1)
        connections = q @ k.transpose(1, 2)  # (N, L, S)
        # forward_scan = scan_forward(connections)
        backward_scan = scan_backward(connections)
        # eye = torch.eye(L, S, device=q.device)
        # step_forward = eye.roll(1, -1)
        # step_backward = eye.roll(-1, -1)
        # step_forward[-1, 0] = 0
        # step_forward[-1, -1] = 1
        # step_backward[0, -1] = 0
        # step_backward[0, 0] = 1
        # softmax = self.linear(q).softmax(-1)

        return backward_scan


class TransformerEncoderLayer(transformer.TransformerEncoderLayer):
    def build_multihead_attention(self, d_model, dropout, nhead):
        return MultiheadAttention(d_model, nhead, dropout=dropout)


class TransformerDecoderLayer(transformer.TransformerDecoderLayer):
    def build_multihead_attention(self, d_model, dropout, nhead):
        return MultiheadAttention(d_model, nhead, dropout=dropout)


class Transformer(transformer.Transformer):
    @staticmethod
    def build_transformer_decoder_layer(
        activation, d_model, dim_feedforward, dropout, nhead
    ):
        return TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )

    @staticmethod
    def build_transformer_encoder_layer(
        activation, d_model, dim_feedforward, dropout, nhead
    ):
        return TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )


class TransformerModel(models.TransformerModel):
    @staticmethod
    def build_transformer_encoder_layer(dropout, nhead, nhid, ninp):
        return TransformerEncoderLayer(ninp, nhead, nhid, dropout)

    def encode_pos(self, src):
        return src
