import torch
from torch import Tensor

import models
import multihead_attention
import transformer


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
    x = torch.cat([row.roll(-i, -1) for i, row in enumerate(x.split(1, -2))], -2)
    x = scan(x)
    return torch.cat([row.roll(i, -1) for i, row in enumerate(x.split(1, -2))], -2)


def scan_backward(x: Tensor, last_col_1=False) -> Tensor:
    *_, n = x.shape
    x = x.flip(-1)
    if last_col_1:
        x = with_last_col_1(x)
    n -= 1
    x = torch.cat([row.roll(i - n, -1) for i, row in enumerate(x.split(1, -2))], -2)
    x = scan(x)
    x = torch.cat([row.roll(n - i, -1) for i, row in enumerate(x.split(1, -2))], -2)
    return x.flip(-1)


class MultiheadAttention(multihead_attention.MultiheadAttention):
    def __init__(
        self, embed_dim, num_heads, last_col_1: bool, forward_scan: bool, **kwargs
    ):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.forward_scan = forward_scan
        self.last_col_1 = last_col_1

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
        if self.forward_scan:
            assert q.size(-1) == k.size(-1)
            E = q.size(-1)
            q1, q2 = torch.split(q, [E - E // 2, E // 2], -1)
            k1, k2 = torch.split(q, [E - E // 2, E // 2], -1)
            forward_connections = q1 @ k1.transpose(1, 2)  # (N, L, S)
            backward_connections = q2 @ k2.transpose(1, 2)  # (N, L, S)
            forward_scan = scan_forward(backward_connections.sigmoid())
            backward_scan = scan_backward(forward_connections.sigmoid())
            backward_sums = backward_scan.sum(-1, keepdim=True)
            forward_sums = forward_scan.sum(-1, keepdim=True)
            attn_output = (backward_scan + forward_scan) / (
                backward_sums + forward_sums
            )
            return attn_output
        connections = q @ k.transpose(1, 2)  # (N, L, S)
        return scan_backward(connections.sigmoid(), last_col_1=self.last_col_1)

    @staticmethod
    def softmax(attn_output_weights):
        return attn_output_weights

    @staticmethod
    def apply_mask(attn_mask, attn_output_weights):
        return attn_output_weights * attn_mask.exp()


class TransformerEncoderLayer(transformer.TransformerEncoderLayer):
    def build_multihead_attention(self, d_model, dropout, nhead, **kwargs):
        return MultiheadAttention(d_model, nhead, dropout=dropout, **kwargs)


class TransformerModel(models.TransformerModel):
    @staticmethod
    def build_transformer_encoder_layer(dropout, nhead, nhid, ninp, **kwargs):
        return TransformerEncoderLayer(ninp, nhead, nhid, dropout, **kwargs)

    def encode_pos(self, src):
        return src
