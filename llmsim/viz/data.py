"""页面所需的数据：逐阶段、逐层、逐卡的传输量与耗时（字节、微秒）。"""
from __future__ import annotations

from ..layers.dispatcher import A2ACost, PermuteCost
from ..runner import SimResult


def _us(times: list[float]) -> list[float]:
    return [t * 1e6 for t in times]


def _stage(a2a: A2ACost, perm: PermuteCost, a2a_first: bool) -> dict:
    """一个阶段每卡依次执行的两步：dispatch 先传输再 permute，combine 先 unpermute 再传输。"""
    steps = [("a2a", a2a.card_time), ("permute", perm.card_time)]
    return {
        "steps": [{"op": op, "us": _us(t)} for op, t in (steps if a2a_first else steps[::-1])],
        "index_us": _us(perm.index_time),
        "card_time_us": _us([a + b for a, b in zip(a2a.card_time, perm.card_time)]),
        "sent": a2a.sent, "recv": a2a.recv, "local": a2a.local,
    }


def page_data(r: SimResult) -> dict:
    sc = r.scenario
    m = sc.model
    return {
        "server": sc.server.name,
        "model": {"name": m.name, "n_layers": m.n_layers, "n_experts": m.n_experts, "topk": m.topk, "hidden": m.hidden},
        "ep": sc.parallel.ep,
        "placement": sc.parallel.placement,
        "experts_per_card": sc.experts_per_card,
        "transfer": {"mode": sc.transfer.mode, "concurrency": sc.transfer.concurrency, "permute": sc.transfer.permute},
        "precision": {"dispatch": sc.precision.dispatch, "combine": sc.precision.combine},
        "token_bytes": r.token_bytes,
        "phases": [{
            "name": ph.batch.phase,
            "tokens_per_rank": ph.batch.tokens_per_rank[0],
            "layer_us": ph.layer_time * 1e6,
            "step_ms": ph.time * 1e3,
            "layers": [{
                "layer": l.layer,
                "time_us": l.time * 1e6,
                "imbalance": l.routing.imbalance(),
                "pair": l.routing.pair,
                "dispatch": _stage(l.dispatch, l.permute, a2a_first=True),
                "combine": _stage(l.combine, l.unpermute, a2a_first=False),
            } for l in ph.layers],
        } for ph in r.phases],
    }
