"""场景：服务器 + 模型 + 部署 + 精度 + 传输方式 + 负载，从 TOML 读取，可按点号路径覆盖字段。"""
from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .hardware import Server, load_server
from .config import ModelConfig, load_config
from .routing import ROUTING_METHODS

DTYPE_BYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
PERMUTE_MODES = ("segment", "buffer", "none")


@dataclass(frozen=True)
class Parallel:
    ep: int                         # 一个 EP 组的卡数
    redundant_experts: int = 0      # 每个 EP 组的冗余专家数
    placement: str = "eplb"         # static：按编号连续放置；eplb：按热度均衡放置并复制热点专家
    dedup: str = "rank"             # rank：同一 token 发往同一张卡的多个专家只传一次；none：按专家各传一次

    def __post_init__(self):
        if self.placement not in ("static", "eplb"):
            raise ValueError("placement 只能是 static / eplb")
        if self.dedup not in ("rank", "none"):
            raise ValueError("dedup 只能是 rank / none")


@dataclass(frozen=True)
class Precision:
    dispatch: str = "fp8"
    combine: str = "bf16"
    transfer_scale_block: int = 128  # FP8 / FP4 传输每多少个元素附带一个 fp32 scale；0 表示不计
    expert: str = ""                 # 路由专家权重；空表示用模型 quantization_config，再没有就是 fp8
    expert_scale_block: int = 0      # 专家权重每多少个元素一个 1 字节 scale；0 表示不计
    shared: str = "fp8"              # shared expert 权重

    def __post_init__(self):
        for f in ("dispatch", "combine", "shared"):
            if getattr(self, f) not in DTYPE_BYTES:
                raise ValueError(f"precision.{f} 只能是 {tuple(DTYPE_BYTES)}")
        if self.expert and self.expert not in DTYPE_BYTES:
            raise ValueError(f"precision.expert 只能是 {tuple(DTYPE_BYTES)}")


@dataclass(frozen=True)
class Transfer:
    mode: str = "push"              # push：发送方写对端显存；pull：接收方读对端显存
    concurrency: int = 1            # 每张卡同时进行的对端传输数
    aggregate_gbps: float = 0.0     # 每卡对端传输总带宽上限（GB/s），0 表示不设
    local_copy: bool = True         # 发给本卡专家的 token 是否计一次本地拷贝
    permute: str = "segment"        # device-major 与 expert-major 互转的数据搬运：segment 按段 / buffer 整块 / none 不计
    routing: str = "auto"           # 路由抽样：token / aggregate / auto

    def __post_init__(self):
        if self.mode not in ("push", "pull"):
            raise ValueError("a2a.mode 只能是 push / pull")
        if self.concurrency < 1:
            raise ValueError("a2a.concurrency 至少为 1")
        if self.permute not in PERMUTE_MODES:
            raise ValueError(f"a2a.permute 只能是 {PERMUTE_MODES}")
        if self.routing not in ROUTING_METHODS:
            raise ValueError(f"a2a.routing 只能是 {ROUTING_METHODS}")


@dataclass(frozen=True)
class Workload:
    expert_skew: float = 0.0        # 专家热度的 Zipf 指数；0 表示均匀
    seed: int = 0


@dataclass(frozen=True)
class Scenario:
    server: Server
    model: ModelConfig
    parallel: Parallel
    precision: Precision
    transfer: Transfer
    phases: dict[str, int]          # 阶段名 -> 每张卡每步 dispatch 的 token 数
    workload: Workload

    def __post_init__(self):
        if self.parallel.ep > self.server.n_gpus:
            raise ValueError(f"{self.server.name} 只有 {self.server.n_gpus} 张卡，ep={self.parallel.ep} 超出单机范围")

    @property
    def experts_per_card(self) -> float:
        return (self.model.n_experts + self.parallel.redundant_experts) / self.parallel.ep


def _build(cls, d: dict, section: str):
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"[{section}] 不认识的字段：{sorted(unknown)}")
    return cls(**d)


def scenario_from_dict(d: dict) -> Scenario:
    phases = {k: int(v) for k, v in d.get("phases", {}).items()}
    if not phases:
        raise ValueError("场景需要 [phases]，如 prefill = 8192、decode = 128（每卡每步的 token 数）")
    return Scenario(
        server=load_server(d["server"]),
        model=load_config(d["model"]),
        parallel=_build(Parallel, d.get("parallel", {}), "parallel"),
        precision=_build(Precision, d.get("precision", {}), "precision"),
        transfer=_build(Transfer, d.get("a2a", {}), "a2a"),
        phases=phases,
        workload=_build(Workload, d.get("workload", {}), "workload"),
    )


def load_scenario_dict(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_value(text: str):
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def apply_override(d: dict, key: str, value) -> dict:
    """按点号路径覆盖场景字段，如 apply_override(d, "parallel.ep", "4")。返回新 dict。"""
    out = copy.deepcopy(d)
    node = out
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = _parse_value(value) if isinstance(value, str) else value
    return out
