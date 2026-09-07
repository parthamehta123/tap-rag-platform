#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cp -n .env.example .env || true
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
echo "Ready. Run: source .venv/bin/activate && make ingest && make test"
