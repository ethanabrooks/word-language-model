from typing import Optional, Tuple

import torch
import multihead_attention
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


def self_cat(x: Tensor) -> Tensor:
    *_, n, d = x.shape
    return torch.cat(
        [x.unsqueeze(0).expand(n, -1, -1), x.unsqueeze(1).expand(-1, n, -1)], dim=-1
    ).reshape(n ** 2, d * 2)


class MultiheadAttention(multihead_attention.MultiheadAttention):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
    raise NotImplementedError