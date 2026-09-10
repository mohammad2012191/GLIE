#!/usr/bin/env bash
# Table 4: Light-ColPali's fine-tuning stage, trained on the same source pages GLIE fits on and
# evaluated zero-shot on all queries of the ten ViDoRe v1 subsets.
source "$(dirname "$0")/_common.sh"

# 1. sanity check: the untrained merge must reproduce stock ColPali + merging
python -m glie.train_lightcolpali --datasets $VIDORE_V1 --eval_all_queries --codes $CODES \
    --cache_dir "$CACHE" --out_dir "$OUT" --eval_at_init --tag lcp_init 2>&1 | tee "$OUT/logs/lcp_init.log"

# 2. the fine-tune: 4,000 query-page pairs, five epochs, one LoRA per budget
python -m glie.train_lightcolpali --datasets $VIDORE_V1 --eval_all_queries --codes $CODES \
    --max_train_pairs 4000 --epochs 5 --batch 8 --grad_accum 4 --lr 5e-5 \
    --cache_dir "$CACHE" --out_dir "$OUT" --tag lcp 2>&1 | tee "$OUT/logs/lcp.log"
