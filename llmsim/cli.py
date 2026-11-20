"""命令行入口。

    python -m llmsim configs/scenarios/deepseek_v41_s3.toml
    python -m llmsim <scenario> --html out/moe.html
    python -m llmsim <scenario> --set a2a.concurrency=4 --show-curves
    python -m llmsim <scenario> --sweep parallel.ep=2,4,8 --sweep a2a.concurrency=1,7

--sweep 可重复，多个 --sweep 取所有组合。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

from .hardware import load_server
from .report import format_curves, format_result, format_sweep, result_to_dict
from .runner import simulate
from .scenario import apply_override, load_scenario_dict, scenario_from_dict
from .viz import write_html


def _split(item: str) -> tuple[str, str]:
    key, _, val = item.partition("=")
    return key.strip(), val.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llmsim", description="MoE 层 EP all-to-all 传输时间仿真")
    ap.add_argument("scenario", help="场景 TOML 文件")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖场景字段，点号路径，可重复")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2,...", help="扫描一个字段，可重复")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--html", metavar="PATH", help="另外生成可交互的 HTML 图表（不能与 --sweep 同用）")
    ap.add_argument("--show-curves", action="store_true", help="先打印服务器各访存路径的带宽")
    args = ap.parse_args(argv)
    if args.html and args.sweep:
        ap.error("--html 只用于单次仿真，不能与 --sweep 同用")
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    base = load_scenario_dict(args.scenario)
    for key, val in map(_split, args.set):
        base = apply_override(base, key, val)
    if args.show_curves:
        print(format_curves(load_server(base["server"])), end="\n\n")

    if not args.sweep:
        r = simulate(scenario_from_dict(base))
        print(json.dumps(result_to_dict(r), ensure_ascii=False, indent=2) if args.json else format_result(r))
        if args.html:
            write_html(r, args.html)
            print(f"\nHTML 已写入 {args.html}", file=sys.stderr)
        return 0

    axes = [(key, val.split(",")) for key, val in map(_split, args.sweep)]
    results = []
    for combo in itertools.product(*(vals for _, vals in axes)):
        d = base
        for (key, _), v in zip(axes, combo):
            d = apply_override(d, key, v.strip())
        results.append((",".join(combo), simulate(scenario_from_dict(d))))
    if args.json:
        print(json.dumps([result_to_dict(r) for _, r in results], ensure_ascii=False, indent=2))
    else:
        print(format_sweep(",".join(key.split(".")[-1] for key, _ in axes), results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
