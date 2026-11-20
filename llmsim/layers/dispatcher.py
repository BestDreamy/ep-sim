"""Token dispatcher：MoE 层里 token 的卡间分发与顺序变换，只计开销、不搬运数据。

    dispatch   all-to-all：token 连同路由元数据（topk 专家 id 与权重）发往专家所在卡，
               收到的 token 按来源卡排列（device-major）
    permute    device-major -> expert-major：专家计算要求每个本地专家的 token 连续存放
    unpermute  expert-major -> device-major：专家输出按目标卡分段，准备发回
    combine    all-to-all：专家输出发回 token 所在卡

all-to-all：push 是发送卡写对端显存，pull 是接收卡读对端显存，发往本卡专家的部分走本卡读写；
每卡 concurrency 条通道并发，耗时取发送侧与接收侧的较大者。

顺序变换分两步，都是本卡显存的读写：
    index  算出新顺序：读收到 token 的 topk 专家 id，写每个本地专家的 token 数（前缀和得到偏移）
           和每一行的置换索引；unpermute 复用这份索引，只需读一次
    data   按置换索引搬运 token 数据。segment：每个来源卡 / 每个本地专家各一段访存；
           buffer：读写各整块一次；none：不计
"""
from __future__ import annotations

from dataclasses import dataclass

from ..hardware import Server
from ..routing import RoutingStats
from ..scenario import DTYPE_BYTES, Precision, Transfer

INDEX_BYTES = 4                     # int32：专家 id、计数、置换索引
TOPK_META_BYTES = INDEX_BYTES + DTYPE_BYTES["fp32"]   # 每个 topk 项随 token 发送的专家 id + 权重


@dataclass
class A2ACost:
    """一次 all-to-all。时间单位秒，大小单位字节，列表按卡索引。"""
    card_time: list[float]
    sent: list[float]               # 发往对端的字节
    recv: list[float]               # 从对端收到的字节
    local: list[float]              # 发往本卡专家的字节
    payloads: list[float]           # 每次对端传输的大小

    @property
    def peer_payload_mean(self) -> float:
        return sum(self.payloads) / len(self.payloads) if self.payloads else 0.0


@dataclass
class PermuteCost:
    """一次顺序变换，列表按卡索引（秒）。"""
    index_time: list[float]         # 算出新顺序
    data_time: list[float]          # 搬运 token 数据

    @property
    def card_time(self) -> list[float]:
        return [a + b for a, b in zip(self.index_time, self.data_time)]


def token_bytes(hidden: int, dtype: str, scale_block: int) -> float:
    b = hidden * DTYPE_BYTES[dtype]
    if dtype in ("fp8", "fp4") and scale_block:
        b += hidden / scale_block * DTYPE_BYTES["fp32"]
    return b


def _lanes(times: list[float], k: int) -> float:
    """把若干次传输排到 k 条并发通道上（最长优先），返回最晚完成时间。"""
    if k == 1:
        return sum(times)
    if len(times) <= k:
        return max(times, default=0.0)
    lanes = [0.0] * k
    for t in sorted(times, reverse=True):
        lanes[lanes.index(min(lanes))] += t
    return max(lanes)


class TokenDispatcher:
    def __init__(self, server: Server, transfer: Transfer, precision: Precision, hidden: int, topk: int,
                 n_local_experts: int):
        self.server = server
        self.transfer = transfer
        self.topk = topk
        self.n_local_experts = n_local_experts
        self.hidden_bytes = token_bytes(hidden, precision.dispatch, precision.transfer_scale_block)
        self.dispatch_bytes = self.hidden_bytes + topk * TOPK_META_BYTES
        self.combine_bytes = token_bytes(hidden, precision.combine, precision.transfer_scale_block)

    def dispatch(self, r: RoutingStats) -> A2ACost:
        return self._all_to_all(r.pair, self.dispatch_bytes)

    def permute(self, r: RoutingStats) -> PermuteCost:
        load = self.server.gpu.local.load
        store = self.server.gpu.local.store
        index = [load.time(sum(src) * self.topk * INDEX_BYTES)
                 + store.time(self.n_local_experts * INDEX_BYTES)
                 + store.time(sum(rows) * INDEX_BYTES)
                 for src, rows in zip(_by_source(r), _by_expert(r))]
        data = [self._move(src, rows, self.hidden_bytes) for src, rows in zip(_by_source(r), _by_expert(r))]
        return PermuteCost(index, data)

    def unpermute(self, r: RoutingStats) -> PermuteCost:
        load = self.server.gpu.local.load
        index = [load.time(sum(rows) * INDEX_BYTES) for rows in _by_expert(r)]
        data = [self._move(rows, dst, self.combine_bytes) for dst, rows in zip(_by_source(r), _by_expert(r))]
        return PermuteCost(index, data)

    def combine(self, r: RoutingStats) -> A2ACost:
        return self._all_to_all([list(col) for col in zip(*r.pair)], self.combine_bytes)

    def _move(self, read_counts: list[int], write_counts: list[int], bytes_per_token: float) -> float:
        """按 read_counts 分段读出本卡显存，再按 write_counts 分段写回。"""
        mode = self.transfer.permute
        if mode == "none":
            return 0.0
        if mode == "buffer":
            read_counts, write_counts = [sum(read_counts)], [sum(write_counts)]
        local = self.server.gpu.local
        return (sum(local.load.time(n * bytes_per_token) for n in read_counts)
                + sum(local.store.time(n * bytes_per_token) for n in write_counts))

    def _all_to_all(self, pair: list[list[int]], bytes_per_token: float) -> A2ACost:
        """pair[s][d]：卡 s 要发给卡 d 的 token 份数。"""
        tr, server, ep = self.transfer, self.server, len(pair)
        op = "store" if tr.mode == "push" else "load"
        send: list[list[float]] = [[] for _ in range(ep)]
        recv: list[list[float]] = [[] for _ in range(ep)]
        cost = A2ACost([0.0] * ep, [0.0] * ep, [0.0] * ep, [0.0] * ep, [])
        for s in range(ep):
            for d in range(ep):
                size = pair[s][d] * bytes_per_token
                if not size:
                    continue
                if s == d:
                    cost.local[s] = size
                    if tr.local_copy:
                        send[s].append(server.local_time(op, size))
                    continue
                # push：卡 s 写卡 d 的显存；pull：卡 d 读卡 s 的显存
                t = server.peer_time(op, s, d, size) if tr.mode == "push" else server.peer_time(op, d, s, size)
                send[s].append(t)
                recv[d].append(t)
                cost.sent[s] += size
                cost.recv[d] += size
                cost.payloads.append(size)

        cap = tr.aggregate_gbps * 1e9
        for c in range(ep):
            ts, tv = _lanes(send[c], tr.concurrency), _lanes(recv[c], tr.concurrency)
            if cap:
                ts, tv = max(ts, cost.sent[c] / cap), max(tv, cost.recv[c] / cap)
            cost.card_time[c] = max(ts, tv)
        return cost


def _by_source(r: RoutingStats) -> list[list[int]]:
    """卡 d 从各卡收到的 token 份数（device-major 的分段）。"""
    return [[row[d] for row in r.pair] for d in range(len(r.pair))]


def _by_expert(r: RoutingStats) -> list[list[int]]:
    """卡 d 上各本地专家的 token 数（expert-major 的分段）。"""
    return [list(t.values()) for t in r.expert_tokens]
