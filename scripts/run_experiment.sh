#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/example.yaml}"
STAMP="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
OUTDIR="results/${STAMP}"
mkdir -p "$OUTDIR"
cp "$CONFIG" "$OUTDIR/config.yaml"
uv run python scripts/run_experiment.py --config "$CONFIG" --outdir "$OUTDIR"
echo "Results in $OUTDIR"
