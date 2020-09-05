from typing import Optional, Tuple

import torch
from torch.nn import Parameter

import model
import multihead_attention
import transformer
from torch import Tensor


def with_last_col_1(x: Tensor, last1=None):
    if last1 is None:
        last1 = torch.zeros_like(x)
        last1[..., -1] = 1
    # noinspection PyTypeChecker
    return last1 * (1 - x) + (1 - last1) * x


def scan(x: Tensor) -> Tensor:
    # noinspection PyTypeChecker,PyUnresolvedReferences
    return with_last_col_1((1 - x).cumprod(-1)).roll(1, -1) * x


def scan_in_time(x: Tensor) -> Tensor:
    x = torch.cat([row.roll(-i, -1) for i, row in enumerate(x.split(1, -2))], -2)
    x = scan(with_last_col_1(x))
    return torch.cat([row.roll(i, -1) for i, row in enumerate(x.split(1, -2))], -2)


class MultiheadAttention(multihead_attention.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__(embed_dim, num_heads, *args, **kwargs)
        head_dim = embed_dim // num_heads
        self.connect_proj_weight = Parameter(torch.Tensor(head_dim, head_dim))

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
        connections = q @ self.connect_proj_weight @ k.transpose(1, 2)  # (N, L, S)
        # TODO: add backward scan, 1 step forward, 1 step backward.
        return scan_in_time(connections)


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


class TransformerModel(model.TransformerModel):
    @staticmethod
    def build_transformer_encoder_layer(dropout, nhead, nhid, ninp):
        return TransformerEncoderLayer(ninp, nhead, nhid, dropout)
