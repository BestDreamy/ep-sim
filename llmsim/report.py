"""文本 / JSON 报告。"""
from __future__ import annotations

from dataclasses import asdict

from .hardware import OPS, Access, PayloadCurve, Server
from .model import ForwardCost
from .runner import SimResult

GB = 1e9


def _size(b: float) -> str:
    return f"{b / 1024 ** 2:.2f}MiB" if b >= 1024 ** 2 else f"{b / 1024:.1f}KiB"


def _us(t: float) -> str:
    return f"{t * 1e6:.1f} us"


def _format_access(title: str, a: Access) -> list[str]:
    """曲线：每段的幂指数 k 与若干 payload 上的拟合带宽（* 为插值 / 外推点）；固定带宽：一行。"""
    if not isinstance(a.load, PayloadCurve):
        return [f"== {title}：load {a.load.gbps:g} GB/s + {a.load.latency_us:g} us，"
                f"store {a.store.gbps:g} GB/s + {a.store.latency_us:g} us =="]
    sizes = a.load.sizes
    lines = [f"== {title}（GB/s），分段幂函数 bw = bw_i·(s/s_i)^k ==",
             f"{'segment':<21}" + "".join(f"{op + ' k':>12}" for op in OPS)]
    for i in range(len(sizes) - 1):
        seg = f"{_size(sizes[i])} - {_size(sizes[i + 1])}"
        lines.append(f"{seg:<21}" + "".join(f"{getattr(a, op).exponent(i):12.3f}" for op in OPS))
    lines.append(f"{'payload':<21}" + "".join(f"{op:>12}" for op in OPS))
    for s in sorted({*sizes, *(x * 2 for x in sizes), sizes[0] / 2}):
        label = _size(s) + ("" if s in sizes else " *")
        lines.append(f"{label:<21}" + "".join(f"{getattr(a, op).bandwidth(s) / 1e9:12.2f}" for op in OPS))
    return lines


def format_curves(server: Server) -> str:
    links = sorted({link for row in server.topology for link in row} - {"-"})
    lines = [f"服务器 {server.name}：{server.n_gpus} × {server.gpu.name}，卡间链路 {', '.join(links)}", ""]
    lines += _format_access(f"{server.gpu.name} 本卡显存", server.gpu.local) + [""]
    for name in links:
        lines += _format_access(f"链路 {name}", server.links[name]) + [""]
    lines += (_format_access(f"NIC {server.nic.name}（不参与仿真）", server.nic.access) if server.nic
              else ["NIC：未配置"])
    return "\n".join(lines)


def _mean(ph: ForwardCost, f) -> float:
    """f(layer) 在各层上的平均值。"""
    return sum(f(l) for l in ph.layers) / len(ph.layers)


def _stage_line(ph: ForwardCost, name: str, a2a, perm, ep: int) -> str:
    """一个阶段：各层最慢卡耗时的平均，以及其中各步（各自最慢卡）的平均。"""
    stage = _mean(ph, lambda l: max(getattr(l, f"{name}_stage")))
    t_a2a = _mean(ph, lambda l: max(a2a(l).card_time))
    t_perm = _mean(ph, lambda l: max(perm(l).card_time))
    t_index = _mean(ph, lambda l: max(perm(l).index_time))
    sent = _mean(ph, lambda l: sum(a2a(l).sent)) / ep
    payload = _mean(ph, lambda l: a2a(l).peer_payload_mean)
    order = "unpermute" if name == "combine" else "permute"
    return (f"  {name:<8} 每层 max {_us(stage):>11} = 传输 {_us(t_a2a)} + {order} {_us(t_perm)}"
            f"（其中算顺序 {_us(t_index)}）  payload avg {_size(payload)}  每卡每层发往对端 {_size(sent)}")


def format_result(r: SimResult) -> str:
    sc = r.scenario
    server, m, p, prec, tr = sc.server, sc.model, sc.parallel, sc.precision, sc.transfer
    params = sum(r.param_bytes.values())
    hbm = server.gpu.hbm_gb * 1e9
    lines = [
        f"== {m.name} MoE 层通信 on {server.name}（只算传输与顺序变换，不含专家计算） ==",
        f"EP={p.ep}  experts={m.n_experts}+{p.redundant_experts} ({p.placement})  topk={m.topk}  "
        f"hidden={m.hidden}  MoE layers={m.n_moe_layers}",
        f"mode={tr.mode}  concurrency={tr.concurrency}  permute={tr.permute}  dedup={p.dedup}  "
        f"dispatch={prec.dispatch}  combine={prec.combine}  expert_skew={sc.workload.expert_skew}",
        f"MoE 权重/卡: {params / GB:.1f} GB（路由专家 {sc.experts_per_card:g} 个 "
        f"{r.param_bytes['routed_experts'] / GB:.1f} GB + shared {r.param_bytes['shared_experts'] / GB:.1f} GB），"
        f"HBM {hbm / GB:.0f} GB 剩 {(hbm - params) / GB:.1f} GB",
        "",
    ]
    for ph in r.phases:
        slowest = max(ph.layers, key=lambda l: l.time)
        lines += [
            f"[{ph.batch.phase}] {ph.batch.tokens_per_rank[0]} tokens/卡/步  routing={ph.layers[0].routing.method}  "
            f"每张卡经过 {len(ph.layers)} 个 MoE 层  负载不均衡 avg {_mean(ph, lambda l: l.routing.imbalance()):.2f}",
            _stage_line(ph, "dispatch", lambda l: l.dispatch, lambda l: l.permute, p.ep),
            _stage_line(ph, "combine", lambda l: l.combine, lambda l: l.unpermute, p.ep),
            f"  每层 avg {_us(ph.layer_time)}（最慢 L{slowest.layer} {_us(slowest.time)}）  每步 {ph.time * 1e3:.2f} ms",
            "",
        ]
    lines.append("每个阶段取最慢的卡；等号右边各步的数值是各自最慢卡的平均，相加不一定等于阶段耗时")
    return "\n".join(lines)


def format_sweep(key: str, results: list[tuple[str, SimResult]]) -> str:
    names = [ph.batch.phase for ph in results[0][1].phases]
    width = max(len(key), *(len(v) for v, _ in results))
    head = f"{key:>{width}} | {'MoE GB':>6} |" + "|".join(
        f" {n + ' disp/comb us':>20} | {'payload':>9} | {'step ms':>8} " for n in names)
    lines = [head, "-" * len(head)]
    for val, r in results:
        cells = []
        for ph in r.phases:
            dc = (f"{_mean(ph, lambda l: max(l.dispatch_stage)) * 1e6:.1f}/"
                  f"{_mean(ph, lambda l: max(l.combine_stage)) * 1e6:.1f}")
            payload = _mean(ph, lambda l: l.dispatch.peer_payload_mean)
            cells.append(f" {dc:>20} | {_size(payload):>9} | {ph.time * 1e3:8.2f} ")
        lines.append(f"{val:>{width}} | {sum(r.param_bytes.values()) / GB:6.1f} |" + "|".join(cells))
    lines.append("MoE GB = 每卡 MoE 权重；disp/comb = 各层 dispatch / combine 阶段最慢卡耗时的平均（含顺序变换）；"
                 "payload = dispatch 单次对端传输平均大小；step = 所有 MoE 层合计")
    return "\n".join(lines)


def result_to_dict(r: SimResult) -> dict:
    sc = r.scenario
    return {
        "server": {"name": sc.server.name, "gpu": sc.server.gpu.name, "n_gpus": sc.server.n_gpus},
        "model": {**asdict(sc.model), "n_moe_layers": sc.model.n_moe_layers},
        "parallel": asdict(sc.parallel),
        "precision": asdict(sc.precision),
        "a2a": asdict(sc.transfer),
        "workload": asdict(sc.workload),
        "param_gb": {k: v / GB for k, v in r.param_bytes.items()},
        "phases": [{
            "name": ph.batch.phase,
            "tokens_per_rank": ph.batch.tokens_per_rank[0],
            "routing": ph.layers[0].routing.method,
            "layer_us": ph.layer_time * 1e6,
            "step_ms": ph.time * 1e3,
            "layers": [{
                "layer": l.layer,
                "dispatch_us": max(l.dispatch_stage) * 1e6,
                "permute_us": max(l.permute.card_time) * 1e6,
                "permute_index_us": max(l.permute.index_time) * 1e6,
                "combine_us": max(l.combine_stage) * 1e6,
                "unpermute_us": max(l.unpermute.card_time) * 1e6,
                "imbalance": l.routing.imbalance(),
            } for l in ph.layers],
        } for ph in r.phases],
    }
