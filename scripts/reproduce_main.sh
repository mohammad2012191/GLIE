#!/usr/bin/env bash
# Table 2 (ViDoRe v1, all methods, all budgets) and the component ladder, three seeds.
# The codec is fitted on 5,000 pages of the ColPali training collection and applied zero-shot.
source "$(dirname "$0")/_common.sh"

for SEED in 0 1 2; do
  TAG=main; [ "$SEED" != 0 ] && TAG=main_seed$SEED
  python -m glie.run_main --datasets $VIDORE_V1 $STD --train_on_max_pages 5000 \
      --codes $CODES --seed $SEED --cache_dir "$CACHE" --out_dir "$OUT" --tag $TAG \
      2>&1 | tee "$OUT/logs/$TAG.log"
done

python - <<'EOF'
import pandas as pd, os
R=os.environ.get("OUT","./out")+"/results"
d=pd.concat([pd.read_csv(f"{R}/{t}__main_table.csv") for t in ["main","main_seed1","main_seed2"]
             if os.path.exists(f"{R}/{t}__main_table.csv")])
c=[x for x in d.columns if x in ("dataset","slug","corpus")][0]
ks=sorted(k for k in d.k.unique() if k<1000)
rows=["raw_kmeans","token_pool","cluster_merge","norm_kmeans","glie_stage1","glie_decoder","glie_oracle"]
piv=d[d.method!="ceiling"].pivot_table(index="method",columns="k",values="ndcg@5",aggfunc="mean").reindex(rows)
print("\nViDoRe v1, macro nDCG@5 over all queries, mean over available seeds\n")
print(piv[ks].round(3).to_string()); print(f"\nuncompressed = {d[d.method=='ceiling']['ndcg@5'].mean():.3f}")
EOF
