"""MoE 层：router → dispatch → permute → experts → unpermute → combine，forward 时只计开销。"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..batch import Batch
from ..routing import (RoutingStats, expert_popularity, place_experts, resolve_method, sample_aggregate,
                       sample_by_token)
from ..scenario import DTYPE_BYTES, Scenario
from .dispatcher import A2ACost, PermuteCost, TokenDispatcher


@dataclass
class LayerCost:
    layer: int                      # 模型中的层号
    routing: RoutingStats
    dispatch: A2ACost
    permute: PermuteCost
    unpermute: PermuteCost
    combine: A2ACost

    @property
    def dispatch_stage(self) -> list[float]:
        """每卡：收发 token，再 permute。"""
        return [a + b for a, b in zip(self.dispatch.card_time, self.permute.card_time)]

    @property
    def combine_stage(self) -> list[float]:
        """每卡：unpermute，再收发专家输出。"""
        return [a + b for a, b in zip(self.unpermute.card_time, self.combine.card_time)]

    @property
    def time(self) -> float:
        """两次 all-to-all 各是一次全卡同步，每个阶段取最慢的卡。"""
        return max(self.dispatch_stage) + max(self.combine_stage)


class Router:
    """代替 gate 计算：按本层的专家热度为每个 token 选 topk 个专家（抽样），再映射到专家副本所在的卡。"""

    def __init__(self, sc: Scenario, rng: random.Random):
        cfg, p = sc.model, sc.parallel
        self.topk = cfg.topk
        self.dedup = p.dedup
        self.method = sc.transfer.routing
        self.rng = rng
        self.popularity = expert_popularity(cfg.n_experts, sc.workload.expert_skew, rng)
        self.placement = place_experts(self.popularity, p.ep, p.redundant_experts, p.placement)

    def __call__(self, batch: Batch) -> RoutingStats:
        tokens = list(batch.tokens_per_rank)
        if resolve_method(self.method, sum(tokens), self.topk) == "token":
            return sample_by_token(self.placement, self.popularity, self.topk, tokens, self.dedup, self.rng)
        return sample_aggregate(self.placement, self.popularity, self.topk, tokens, self.dedup, self.rng)


class MoELayer:
    def __init__(self, layer_id: int, sc: Scenario, rng: random.Random):
        cfg, prec = sc.model, sc.precision
        self.layer_id = layer_id
        self.router = Router(sc, rng)
        self.dispatcher = TokenDispatcher(sc.server, sc.transfer, prec, cfg.hidden, cfg.topk,
                                          len(self.router.placement.slots[0]))
        per_expert = 3 * cfg.hidden * cfg.intermediate          # gate / up / down
        expert_dtype = prec.expert or cfg.expert_dtype or "fp8"
        expert_bytes = DTYPE_BYTES[expert_dtype] + (1 / prec.expert_scale_block if prec.expert_scale_block else 0)
        self.params = {                                           # 每卡常驻的权重字节数
            "routed_experts": sc.experts_per_card * per_expert * expert_bytes,
            "shared_experts": cfg.n_shared * per_expert * DTYPE_BYTES[prec.shared],
        }

    def forward(self, batch: Batch) -> LayerCost:
        routing = self.router(batch)
        dispatch = self.dispatcher.dispatch(routing)
        permute = self.dispatcher.permute(routing)
        # experts：专家计算的开销暂不计
        unpermute = self.dispatcher.unpermute(routing)
        combine = self.dispatcher.combine(routing)
        return LayerCost(self.layer_id, routing, dispatch, permute, unpermute, combine)
