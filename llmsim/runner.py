"""运行一个场景：构建模型，对每个阶段的 batch 做一次 forward。"""
from __future__ import annotations

from dataclasses import dataclass

from .batch import Batch
from .model import ForwardCost, Transformer
from .scenario import Scenario


@dataclass
class SimResult:
    scenario: Scenario
    param_bytes: dict[str, float]   # 每卡常驻的权重字节数
    token_bytes: dict[str, float]   # 每份 token 的传输字节数（dispatch 含路由元数据）
    phases: list[ForwardCost]


def simulate(sc: Scenario) -> SimResult:
    model = Transformer(sc)
    dispatcher = model.layers[0].dispatcher
    token_bytes = {"dispatch": dispatcher.dispatch_bytes, "combine": dispatcher.combine_bytes}
    batches = [Batch(name, (tokens,) * sc.parallel.ep) for name, tokens in sc.phases.items()]
    return SimResult(sc, model.param_bytes(), token_bytes, [model.forward(b) for b in batches])
