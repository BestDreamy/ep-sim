"""模型配置，从 HuggingFace 风格的 config.json 解析（对应 transformers 的 PretrainedConfig），
只取 MoE 层相关的参数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .hardware import CONFIG_DIR


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    n_layers: int
    n_experts: int                  # 路由专家数
    topk: int
    intermediate: int               # 每个专家的 FFN 中间维度
    n_shared: int = 0               # shared expert 数
    first_k_dense: int = 0          # 前 k 层是 dense FFN
    layer_freq: int = 1             # 每隔几层一个 MoE 层
    expert_dtype: str | None = None  # quantization_config.expert_dtype

    @property
    def moe_layer_ids(self) -> list[int]:
        return list(range(self.first_k_dense, self.n_layers, self.layer_freq))

    @property
    def n_moe_layers(self) -> int:
        return len(self.moe_layer_ids)


def _first(d: dict, *keys, default=None):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def config_from_hf(cfg: dict, name: str) -> ModelConfig:
    quant = cfg.get("quantization_config") or {}
    cfg = cfg.get("text_config", cfg)
    n_experts = _first(cfg, "n_routed_experts", "num_experts", "num_local_experts")
    if not n_experts:
        raise ValueError(f"{name} 不是 MoE 模型")
    return ModelConfig(
        name=name,
        hidden=cfg["hidden_size"],
        n_layers=cfg["num_hidden_layers"],
        n_experts=n_experts,
        topk=_first(cfg, "num_experts_per_tok", "moe_topk", "top_k"),
        intermediate=_first(cfg, "moe_intermediate_size", "intermediate_size"),
        n_shared=_first(cfg, "n_shared_experts", "num_shared_experts", default=0),
        first_k_dense=_first(cfg, "first_k_dense_replace", default=0),
        layer_freq=_first(cfg, "moe_layer_freq", "decoder_sparse_step", default=1),
        expert_dtype=quant.get("expert_dtype"),
    )


def load_config(name_or_path: str) -> ModelConfig:
    """接受内置名（如 "deepseek_v4_1"）、config.json 路径或含 config.json 的目录。"""
    path = Path(name_or_path)
    if path.is_dir():
        path = path / "config.json"
    elif not path.suffix:
        path = CONFIG_DIR / "models" / f"{name_or_path}.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return config_from_hf(cfg, name=cfg.get("_name_or_path") or path.stem)
