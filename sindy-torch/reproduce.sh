#!/usr/bin/env bash
# Reproduce the headline demo: simulate -> discover -> evaluate, clean Lorenz data.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
python3 scripts/experiment_clean.py --system lorenz
