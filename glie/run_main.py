"""Main-results driver: every method, every budget, every dataset, ONE eval harness.

    python -m glie.run_main --datasets vidore/arxivqa_test_subsampled --codes 2 4 8 16 32 64

Produces results/<tag>__results.json and results/<tag>__main_table.csv with these rows per
(dataset, k):

    ceiling         uncompressed
    raw_kmeans      naive
    norm_kmeans     spherical projection, free
    cluster_merge   Light-ColPali's merge, training-free
    token_pool      pooling incumbent
    glie_stage1     ours, single-stage  <- compute-matched to every baseline above
    glie_decoder    ours, two-stage     <- storage-matched, extra query compute (stated)
    glie_oracle     shortlist ceiling, bounds glie_decoder

Storage is reported in BYTES PER PAGE, including GLIE's per-cluster norm and count sidecar, so the
comparison is storage-matched and not vector-count-matched.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .baselines import all_static_baselines
from .config import build_parser, ensure_dirs, PRIMARY_METRIC
from .data import kmeans_artifact, load_corpus, make_split
from .metrics import compression_vs_full, evaluate_static, storage_bytes
from .train_glie import evaluate_transfer, train_multi, train_one


def main():
    args = ensure_dirs(build_parser().parse_args())
    assert not (args.eval_all_queries and not args.train_on), (
        "--eval_all_queries requires --train_on: without it the codec is fitted on the target's "
        "own training queries and evaluating on all queries would score the training set.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  refiner={args.refiner}  decoder={args.decoder}  tag={args.tag}")
    print(f"metric={PRIMARY_METRIC}  split_seed={args.split_seed}  holdout={args.holdout_frac}")

    # TRANSFER MODE: fit the codec once on --train_on, then apply it frozen everywhere else.
    src_ckpts, src_slugs = {}, set()
    if args.train_on:
        print(f"\n### transfer mode: fitting ONE codec on {args.train_on}, "
              f"then applying it frozen to {args.datasets}")
        src_cap = args.train_on_max_pages or args.max_pages
        srcs = [load_corpus(d, args.cache_dir, args.model_name, args.encode_batch,
                            src_cap, device, split=args.train_on_split)
                for d in args.train_on]
        src_slugs = {c.slug for c in srcs}
        suffix = "-".join(sorted(s[:6] for s in src_slugs))
        for k in args.codes:
            pairs = [(c, kmeans_artifact(c, k, args.cache_dir, seed=args.split_seed)) for c in srcs]
            if len(pairs) == 1:
                train_one(pairs[0][0], pairs[0][1], k, args, device, args.out_dir)
                src_ckpts[k] = os.path.join(args.out_dir, "checkpoints",
                                            f"{pairs[0][0].slug}_k{k}_{args.tag}.pt")
            else:
                src_ckpts[k] = train_multi(pairs, k, args, device, args.out_dir, suffix)
        del srcs

    rows, blob = [], {}
    for ds in args.datasets:
        corpus = load_corpus(ds, args.cache_dir, args.model_name, args.encode_batch,
                             args.max_pages, device)
        train_q, hold_q = make_split(corpus.n_queries, args.split_seed, args.holdout_frac)
        if args.eval_all_queries:
            hold_q = np.arange(corpus.n_queries)
            print(f"\n=== {corpus.slug}: {corpus.n_pages} pages, ALL {corpus.n_queries} queries "
                  f"evaluated (standard protocol), P={corpus.n_patches} D={corpus.dim}")
        else:
            print(f"\n=== {corpus.slug}: {corpus.n_pages} pages, {corpus.n_queries} queries "
                  f"({len(train_q)} train / {len(hold_q)} holdout), "
                  f"P={corpus.n_patches} D={corpus.dim}")

        ceil, _ = evaluate_static(corpus.page, corpus.pmask, corpus, hold_q, device)
        print(f"    ceiling  {PRIMARY_METRIC}={ceil[PRIMARY_METRIC]:.4f}")
        assert ceil[PRIMARY_METRIC] > 0.4, (
            f"ceiling is {ceil[PRIMARY_METRIC]:.4f}; the uncompressed encoder should score far "
            f"higher. Cache, alignment, or masks are broken -- refusing to report a sweep whose "
            f"every number would be meaningless.")
        rows.append({"dataset": corpus.slug, "k": corpus.n_patches, "method": "ceiling",
                     **ceil, **storage_bytes(corpus.n_patches, corpus.dim, "full"),
                     "compression": 1.0})

        for k in args.codes:
            print(f"\n  --- {corpus.slug} k={k} ---")
            art = kmeans_artifact(corpus, k, args.cache_dir, seed=args.split_seed)

            for name, (reps, rmask) in all_static_baselines(corpus, art, k, args.cache_dir).items():
                m, _ = evaluate_static(reps, rmask, corpus, hold_q, device)
                rows.append({"dataset": corpus.slug, "k": k, "method": name, **m,
                             **storage_bytes(k, corpus.dim, name),
                             "compression": compression_vs_full(k, corpus.n_patches)})
                print(f"    {name:<16}{PRIMARY_METRIC}={m[PRIMARY_METRIC]:.4f}")

            if args.train_on and corpus.slug not in src_slugs:
                res = evaluate_transfer(src_ckpts[k], corpus, art, k, args, device)
                print(f"    [transfer from {', '.join(sorted(src_slugs))}]")
            else:
                res = train_one(corpus, art, k, args, device, args.out_dir)
            blob[f"{corpus.slug}_k{k}"] = res
            for method, key in [("glie_stage1", "stage1"), ("glie_decoder", "decoder_metrics"),
                                ("glie_oracle", "oracle")]:
                rows.append({"dataset": corpus.slug, "k": k, "method": method, **res[key],
                             **res["storage"], "compression": res["compression_vs_full"]})
            print(f"    {'glie_stage1':<16}{PRIMARY_METRIC}={res['stage1'][PRIMARY_METRIC]:.4f}")
            print(f"    {'glie_decoder':<16}{PRIMARY_METRIC}="
                  f"{res['decoder_metrics'][PRIMARY_METRIC]:.4f}  "
                  f"(oracle {res['oracle'][PRIMARY_METRIC]:.4f}, {res['minutes']:.1f} min)")

            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.out_dir, "results", f"{args.tag}__main_table.csv"),
                      index=False)
            json.dump(blob, open(os.path.join(args.out_dir, "results",
                                              f"{args.tag}__results.json"), "w"), indent=2)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"MAIN TABLE ({PRIMARY_METRIC}), averaged over {len(args.datasets)} dataset(s)")
    print("=" * 96)
    piv = (df[df.method != "ceiling"]
           .pivot_table(index="method", columns="k", values=PRIMARY_METRIC, aggfunc="mean"))
    order = ["raw_kmeans", "token_pool", "cluster_merge", "norm_kmeans",
             "glie_stage1", "glie_decoder", "glie_oracle"]
    piv = piv.reindex([m for m in order if m in piv.index])
    print(piv.round(4).to_string())
    ceil_mean = df[df.method == "ceiling"][PRIMARY_METRIC].mean()
    print(f"\nceiling = {ceil_mean:.4f}")

    # The honest baseline: whichever training-free method is strongest, chosen PER DATASET and PER
    # BUDGET, then the margin is macro-averaged over datasets. Taking the max over the pooled
    # (dataset, method) rows would compare our cross-dataset MEAN against the best baseline on the
    # single easiest dataset -- which is not a margin at all.
    TF = ["norm_kmeans", "cluster_merge", "token_pool"]
    tf = df[df.method.isin(TF)]
    best_tf = tf.groupby(["dataset", "k"])[PRIMARY_METRIC].max()
    winner = tf.loc[tf.groupby(["dataset", "k"])[PRIMARY_METRIC].idxmax()] \
               .set_index(["dataset", "k"])["method"]
    print("\nmargin of glie_decoder over the BEST training-free baseline "
          "(chosen per dataset+budget, macro-averaged over datasets):")
    for row_name, label in [("glie_stage1", "stage1 "), ("glie_decoder", "decoder")]:
        g = df[df.method == row_name].set_index(["dataset", "k"])[PRIMARY_METRIC]
        common = g.index.intersection(best_tf.index)
        margin = (g.loc[common] - best_tf.loc[common]).groupby("k").mean()
        gm = g.loc[common].groupby("k").mean()
        bm = best_tf.loc[common].groupby("k").mean()
        print(f"  [{label}]")
        for k in sorted(margin.index):
            picks = sorted(set(winner.loc[[i for i in common if i[1] == k]].tolist()))
            print(f"    k={k:<4} {gm[k]:.4f} - {bm[k]:.4f} = {margin[k]:+.4f}   "
                  f"(baseline: {', '.join(picks)})")

    piv.to_csv(os.path.join(args.out_dir, "results", f"{args.tag}__pivot.csv"))
    print(f"\nwritten to {os.path.join(args.out_dir, 'results')}")


if __name__ == "__main__":
    main()
