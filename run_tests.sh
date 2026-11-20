#!/usr/bin/env bash
# 运行全部单元测试。用法：bash run_tests.sh [-v]
set -euo pipefail
cd "$(dirname "$0")"
python -m unittest discover -s tests "$@"
