"""专家热度、专家放置（含 EPLB 冗余专家）与路由抽样。

只做计数：每个 token 选哪些专家、落到哪张卡，得到每张卡的专家负载和卡到卡的 token 份数。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

ROUTING_METHODS = ("auto", "token", "aggregate")

# 每层逐 token 抽样的规模上限（token 数 × topk），auto 模式下超过后改用聚合近似
TOKEN_SAMPLING_LIMIT = 50_000


def expert_popularity(n_experts: int, skew: float, rng: random.Random) -> list[float]:
    """Zipf 分布的专家热度（随机打乱编号），skew=0 时为均匀分布。"""
    w = [1.0 / (i + 1) ** skew for i in range(n_experts)]
    rng.shuffle(w)
    s = sum(w)
    return [x / s for x in w]


@dataclass
class Placement:
    ep: int
    replicas: list[list[int]]        # 专家 -> 持有其副本的 rank 列表
    slots: list[list[int]]           # rank -> 本地专家列表


def place_experts(popularity: list[float], ep: int, redundant: int, strategy: str) -> Placement:
    n = len(popularity)
    if (n + redundant) % ep:
        raise ValueError(f"专家数 {n} + 冗余 {redundant} 不能被 ep={ep} 整除")
    per_rank = (n + redundant) // ep

    if strategy == "static":
        if redundant:
            raise ValueError("static 放置不支持冗余专家，请改用 placement = \"eplb\"")
        slots = [list(range(r * per_rank, (r + 1) * per_rank)) for r in range(ep)]
        return Placement(ep, [[e // per_rank] for e in range(n)], slots)

    # EPLB：先把冗余副本分给“单副本负载”最高的专家，再按负载从大到小装箱到最空闲的 rank
    n_rep = [1] * n
    for _ in range(redundant):
        hot = max(range(n), key=lambda e: popularity[e] / n_rep[e])
        n_rep[hot] += 1
    items = sorted(((popularity[e] / n_rep[e], e) for e in range(n) for _ in range(n_rep[e])), reverse=True)
    load = [0.0] * ep
    slots: list[list[int]] = [[] for _ in range(ep)]
    replicas: list[list[int]] = [[] for _ in range(n)]
    for w, e in items:
        r = min((r for r in range(ep) if len(slots[r]) < per_rank and e not in slots[r]),
                key=lambda r: load[r])
        slots[r].append(e)
        replicas[e].append(r)
        load[r] += w
    return Placement(ep, replicas, slots)


@dataclass
class RoutingStats:
    method: str                           # 抽样方式：token / aggregate
    expert_tokens: list[dict[int, int]]   # rank -> {专家: token 数}
    pair: list[list[int]]                 # pair[s][d]：rank s 发给 rank d 的 token 份数（含 s == d）

    def imbalance(self) -> float:
        """最忙卡的专家 token 数 / 平均值。"""
        load = [sum(d.values()) for d in self.expert_tokens]
        mean = sum(load) / len(load)
        return max(load) / mean if mean else 1.0


def _empty_stats(method: str, ep: int) -> RoutingStats:
    return RoutingStats(method, [{} for _ in range(ep)], [[0] * ep for _ in range(ep)])


def sample_by_token(placement: Placement, popularity: list[float], topk: int, tokens_per_rank: list[int],
                    dedup: str, rng: random.Random) -> RoutingStats:
    """逐 token 抽样：每个 token 按热度无放回地选 topk 个专家。

    专家有多个副本时优先本卡，否则随机选一个副本。
    dedup = "rank" 时，同一 token 发往同一张卡的多个专家只算一份。
    """
    experts = range(len(popularity))
    cum = []
    acc = 0.0
    for p in popularity:
        acc += p
        cum.append(acc)

    st = _empty_stats("token", placement.ep)
    for s, n_tok in enumerate(tokens_per_rank):
        for _ in range(n_tok):
            chosen: set[int] = set()
            while len(chosen) < topk:
                chosen.update(rng.choices(experts, cum_weights=cum, k=topk - len(chosen)))
            dests = []
            for e in chosen:
                reps = placement.replicas[e]
                if len(reps) == 1:
                    d = reps[0]
                elif s in reps:
                    d = s
                else:
                    d = rng.choice(reps)
                st.expert_tokens[d][e] = st.expert_tokens[d].get(e, 0) + 1
                dests.append(d)
            for d in set(dests) if dedup == "rank" else dests:
                st.pair[s][d] += 1
    return st


def _pick_probabilities(popularity: list[float], topk: int) -> list[float]:
    """每个 token 选中各专家的概率：π_e = 1 - (1 - p_e)^t，t 使 Σπ = topk（逐次无放回抽样的近似）。"""
    def total(t: float) -> float:
        return sum(1.0 - (1.0 - p) ** t for p in popularity)

    lo = hi = float(topk)           # total(topk) ≤ topk，下界成立
    while total(hi) < topk:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) < topk else (lo, mid)
    return [1.0 - (1.0 - p) ** hi for p in popularity]


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    if mean > 30:
        return max(0, round(rng.gauss(mean, mean ** 0.5)))
    limit, k, prod = math.exp(-mean), 0, rng.random()
    while prod > limit:
        k += 1
        prod *= rng.random()
    return k


def _binomial(rng: random.Random, n: int, q: float) -> int:
    if n <= 0 or q <= 0:
        return 0
    if q >= 1:
        return n
    var = n * q * (1 - q)
    if var > 9:
        return min(n, max(0, round(rng.gauss(n * q, var ** 0.5))))
    return min(n, _poisson(rng, n * q))


def sample_aggregate(placement: Placement, popularity: list[float], topk: int, tokens_per_rank: list[int],
                     dedup: str, rng: random.Random) -> RoutingStats:
    """大 token 数时的快速近似（如 prefill）：直接按概率抽 (源卡, 专家副本) 的计数。

    每个专家的多个副本平均分担流量（不做本卡优先）。
    dedup = "rank" 时，rank s 发往 rank d 的份数 ~ Binomial(T_s, q_d)，
    q_d = 1 - Π(1 - π_e / 副本数) 是一个 token 至少命中 d 上一个专家的概率。
    """
    pi = _pick_probabilities(popularity, topk)
    share = [[pi[e] / len(placement.replicas[e]) for e in slots] for slots in placement.slots]
    hit = [1.0 - math.prod(1.0 - min(1.0, x) for x in row) for row in share]

    st = _empty_stats("aggregate", placement.ep)
    for s, n_tok in enumerate(tokens_per_rank):
        for d in range(placement.ep):
            picks = 0
            for e, x in zip(placement.slots[d], share[d]):
                c = _poisson(rng, n_tok * x)
                if c:
                    st.expert_tokens[d][e] = st.expert_tokens[d].get(e, 0) + c
                    picks += c
            st.pair[s][d] = _binomial(rng, n_tok, hit[d]) if dedup == "rank" else picks
    return st


def resolve_method(method: str, total_tokens: int, topk: int) -> str:
    if method not in ROUTING_METHODS:
        raise ValueError(f"routing 只能是 {ROUTING_METHODS}")
    if method == "auto":
        return "token" if total_tokens * topk <= TOKEN_SAMPLING_LIMIT else "aggregate"
    return method
