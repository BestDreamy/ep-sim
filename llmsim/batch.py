"""forward 的输入：只有形状，没有数据。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Batch:
    phase: str                      # prefill / decode
    tokens_per_rank: tuple[int, ...]    # 每张卡上的 token 数
