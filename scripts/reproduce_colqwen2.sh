#!/usr/bin/env bash
# Table 6: the full sweep repeated on ColQwen2 v1.0.
# ColQwen2's adapter loads correctly only with transformers<=4.49; run this in such an environment.
source "$(dirname "$0")/_common.sh"

python -m glie.run_main --datasets $VIDORE_V1 $STD --train_on_max_pages 5000 \
    --model_name vidore/colqwen2-v1.0 --codes $CODES \
    --cache_dir "${CACHE}_colqwen2" --out_dir "$OUT" --tag colqwen2 2>&1 | tee "$OUT/logs/colqwen2.log"
