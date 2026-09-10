#!/usr/bin/env bash
# Table 3: ViDoRe v2, graded multi-relevant nDCG@5, zero-shot with the checkpoints from reproduce_main.sh.
source "$(dirname "$0")/_common.sh"

python -m glie.vidore_v2 --cache_dir "${CACHE}_v2" --ckpt_dir "$OUT/checkpoints" --ckpt_tag main \
    --out_dir "$OUT/results" --codes $CODES --tag v2 2>&1 | tee "$OUT/logs/v2.log"
