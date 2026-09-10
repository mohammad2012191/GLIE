#!/usr/bin/env bash
# Table 1: per-page geometry on the ten ViDoRe v1 subsets, plus the null-model rows.
# CPU only. Requires the caches written by reproduce_main.sh (or any run that extracted the subsets).
source "$(dirname "$0")/_common.sh"

python -m glie.geometry_study   --cache_dir "$CACHE" --out "$OUT/results/geometry.json" \
    2>&1 | tee "$OUT/logs/geometry.log"
python -m glie.geometry_control --cache_dir "$CACHE" --out "$OUT/results/geometry_control.json" \
    --pages 100 2>&1 | tee "$OUT/logs/geometry_control.log"
