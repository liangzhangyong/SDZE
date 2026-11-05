#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --config configs/high_order.json "$@"
