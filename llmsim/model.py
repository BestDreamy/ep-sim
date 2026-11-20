"""Transformer：按模型结构堆叠的层，forward 时逐层计开销。目前只有 MoE 层。"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .batch import Batch
from .layers import LayerCost, MoELayer
from .scenario import Scenario


@dataclass
class ForwardCost:
    batch: Batch
    layers: list[LayerCost] = field(default_factory=list)

    @property
    def time(self) -> float:
        return sum(l.time for l in self.layers)

    @property
    def layer_time(self) -> float:
        return self.time / len(self.layers)


class Transformer:
    def __init__(self, sc: Scenario):
        rng = random.Random(sc.workload.seed)
        self.layers = [MoELayer(i, sc, rng) for i in sc.model.moe_layer_ids]

    def forward(self, batch: Batch) -> ForwardCost:
        cost = ForwardCost(batch)
        for layer in self.layers:
            cost.layers.append(layer.forward(batch))
        return cost

    def param_bytes(self) -> dict[str, float]:
        """每卡常驻的权重字节数，按类别汇总所有层。"""
        total: dict[str, float] = {}
        for layer in self.layers:
            for name, b in layer.params.items():
                total[name] = total.get(name, 0.0) + b
        return total
