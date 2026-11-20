"""HTML 可视化：把一次仿真结果渲染成自包含的交互页面。只读取 SimResult，不参与仿真。

页面由 page.html / page.css / page.js 拼成，数据（data.page_data）以 JSON 内嵌，不依赖外部资源。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..runner import SimResult
from .data import page_data

_DIR = Path(__file__).parent


def _read(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


def render_html(r: SimResult) -> str:
    data = json.dumps(page_data(r), ensure_ascii=False).replace("</", "<\\/")
    page = _read("page.html").replace("/*STYLE*/", _read("page.css")).replace("/*SCRIPT*/", _read("page.js"))
    return page.replace("__DATA__", data)


def write_html(r: SimResult, path: str | Path) -> None:
    Path(path).write_text(render_html(r), encoding="utf-8")
