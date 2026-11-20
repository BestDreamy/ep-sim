"""硬件：服务器（顶层）由若干张 GPU、卡间拓扑和 NIC 组成。

    GPU     本卡显存容量与本卡访存带宽（GPU 自身的特性）
    Server  机内 n×n 拓扑：topology[a][b] 是卡 a 访问卡 b 显存所走的链路，每种链路有自己的读写带宽
    NIC     跨服务器的读写带宽（每卡一个 NIC）；目前只仿真单台服务器，NIC 不参与计算

每种访存路径（Access）都有 load / store 两个方向，带宽是实测的 payload 曲线或固定延迟 + 带宽。
单位：内部统一用秒和字节；配置文件里带宽用 GB/s（1e9 字节/秒），大小用 KiB/MiB（1024 进制）。
"""
from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

OPS = ("load", "store")

_SIZE_UNITS = {"": 1, "b": 1, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}


def parse_size(text: str | int | float) -> float:
    """"4KiB" / "1MiB" / 4096 -> 字节数。"""
    if isinstance(text, (int, float)):
        return float(text)
    m = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]*)\s*", text)
    if not m or m.group(2).lower() not in _SIZE_UNITS:
        raise ValueError(f"无法解析的大小：{text!r}")
    return float(m.group(1)) * _SIZE_UNITS[m.group(2).lower()]


@dataclass(frozen=True)
class PayloadCurve:
    """实测的 payload -> 带宽曲线，拟合成分段幂函数。

    相邻测量点 (s_i, bw_i)、(s_{i+1}, bw_{i+1}) 之间：
        bw(s) = bw_i * (s / s_i) ** k_i,   k_i = log(bw_{i+1} / bw_i) / log(s_{i+1} / s_i)
    即 log-log 坐标下分段线性，曲线经过每个测量点。k_i ≈ 1 表示这段受延迟主导，k_i ≈ 0 表示带宽已饱和。
    小于最小测量点时耗时保持不变（延迟下限）；大于最大测量点时带宽保持最后一个值。
    """
    sizes: tuple[float, ...]        # 字节，严格递增
    gbps: tuple[float, ...]

    def __post_init__(self):
        if len(self.sizes) != len(self.gbps) or len(self.sizes) < 2:
            raise ValueError("带宽曲线至少需要两个点，且 payload 与带宽个数一致")
        if any(b <= a for a, b in zip(self.sizes, self.sizes[1:])):
            raise ValueError("带宽曲线的 payload 必须严格递增")
        if any(g <= 0 for g in self.gbps):
            raise ValueError("带宽必须为正数")

    def exponent(self, i: int) -> float:
        return math.log(self.gbps[i + 1] / self.gbps[i]) / math.log(self.sizes[i + 1] / self.sizes[i])

    def bandwidth(self, size: float) -> float:
        """payload 为 size 字节时的有效带宽，单位字节/秒。"""
        s, g = self.sizes, self.gbps
        if size <= s[0]:
            return g[0] * 1e9 * size / s[0]
        if size >= s[-1]:
            return g[-1] * 1e9
        i = max(j for j in range(len(s) - 1) if s[j] <= size)
        return g[i] * 1e9 * (size / s[i]) ** self.exponent(i)

    def time(self, size: float) -> float:
        return size / self.bandwidth(size) if size > 0 else 0.0


@dataclass(frozen=True)
class AlphaBeta:
    """固定延迟 + 带宽：t = latency + size / bandwidth。"""
    gbps: float
    latency_us: float = 0.0

    def __post_init__(self):
        if self.gbps <= 0:
            raise ValueError("带宽必须为正数")

    def time(self, size: float) -> float:
        return self.latency_us * 1e-6 + size / (self.gbps * 1e9) if size > 0 else 0.0


@dataclass(frozen=True)
class Access:
    """一种访存路径的读（load）、写（store）带宽。"""
    load: PayloadCurve | AlphaBeta
    store: PayloadCurve | AlphaBeta

    def time(self, op: str, size: float) -> float:
        if op not in OPS:
            raise ValueError(f"未知访存操作 {op!r}，可选 {OPS}")
        return getattr(self, op).time(size)


@dataclass(frozen=True)
class GPU:
    name: str
    hbm_gb: float
    local: Access                   # 读写本卡显存
    notes: str = ""


@dataclass(frozen=True)
class NIC:
    name: str
    access: Access                  # load = RDMA read，store = RDMA write
    notes: str = ""


@dataclass(frozen=True)
class Server:
    name: str
    gpu: GPU
    links: dict[str, Access]                # 链路类型 -> 读写带宽
    topology: tuple[tuple[str, ...], ...]   # topology[a][b]：卡 a 访问卡 b 显存所走的链路；对角线为 "-"
    nic: NIC | None = None                  # 跨服务器；目前不参与仿真
    notes: str = ""

    def __post_init__(self):
        n = len(self.topology)
        for a, row in enumerate(self.topology):
            if len(row) != n:
                raise ValueError(f"{self.name} 的 topology 必须是 {n}×{n}")
            for b, link in enumerate(row):
                if (a == b) != (link == "-"):
                    raise ValueError(f"{self.name} topology[{a}][{b}]：对角线写 \"-\"，其余写链路类型")
                if a != b and link not in self.links:
                    raise ValueError(f"{self.name} topology[{a}][{b}] 的链路 {link!r} 没有在 [links] 中定义")

    @property
    def n_gpus(self) -> int:
        return len(self.topology)

    def local_time(self, op: str, size: float) -> float:
        return self.gpu.local.time(op, size)

    def peer_time(self, op: str, src: int, dst: int, size: float) -> float:
        """本机卡 src 对卡 dst 的显存做一次 op。"""
        return self.links[self.topology[src][dst]].time(op, size)


def access_from_dict(d: dict, where: str) -> Access:
    """实测曲线：payload / load / store 三列；或者固定带宽：gbps（+ latency_us），读写相同。"""
    keys = set(d)
    if keys == {"payload", "load", "store"}:
        sizes = tuple(parse_size(s) for s in d["payload"])
        return Access(*(PayloadCurve(sizes, tuple(map(float, d[op]))) for op in OPS))
    if keys in ({"gbps"}, {"gbps", "latency_us"}):
        bw = AlphaBeta(**d)
        return Access(bw, bw)
    raise ValueError(f"{where} 应为 payload/load/store 曲线或 gbps(/latency_us)，实际字段：{sorted(keys)}")


def _load_toml(kind: str, name_or_path: str) -> dict:
    """接受 configs/<kind>/ 下的内置名（如 "s3"）或 TOML 文件路径。"""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / kind / f"{name_or_path.lower()}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_gpu(name_or_path: str) -> GPU:
    d = _load_toml("gpu", name_or_path)
    return GPU(local=access_from_dict(d.pop("local"), "[local]"), **d)


def load_nic(name_or_path: str) -> NIC:
    d = _load_toml("nic", name_or_path)
    return NIC(access=access_from_dict(d.pop("access"), "[access]"), **d)


def load_server(name_or_path: str) -> Server:
    d = _load_toml("server", name_or_path)
    links = {k: access_from_dict(v, f"[links.{k}]") for k, v in d.pop("links").items()}
    return Server(
        gpu=load_gpu(d.pop("gpu")),
        nic=load_nic(d.pop("nic")) if "nic" in d else None,
        links=links,
        topology=tuple(tuple(row) for row in d.pop("topology")),
        **d,
    )
