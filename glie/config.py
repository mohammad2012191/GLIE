"""Single source of truth for every setting in the main-results pipeline.

Anything that could differ between two methods and silently corrupt a comparison lives HERE,
not in an individual script: the split, the metric, the corpus construction, the budgets.
"""
import argparse
import os

VIDORE_V1 = [
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
    "vidore/syntheticDocQA_artificial_intelligence_test",
    "vidore/syntheticDocQA_energy_test",
    "vidore/syntheticDocQA_government_reports_test",
    "vidore/syntheticDocQA_healthcare_industry_test",
]

# The three subsets used for fast iteration / ablations. Full ViDoRe v1 for the main table.
DEV_SUBSETS = VIDORE_V1[:3]

# Primary metric for every table in the paper. ViDoRe's official leaderboard metric is nDCG@5;
# we compute the whole family and report @5 as primary so nobody can accuse us of metric shopping.
PRIMARY_METRIC = "ndcg@5"


def build_parser():
    p = argparse.ArgumentParser()

    # ---- data ----------------------------------------------------------------
    p.add_argument("--datasets", nargs="+", default=DEV_SUBSETS,
                   help="corpora to evaluate on. Without --train_on, the codec is fitted on each "
                        "corpus's own training queries (the in-corpus setting).")
    p.add_argument("--train_on", nargs="+", default=[],
                   help="TRANSFER / ZERO-SHOT MODE. Fit the codec ONCE on these corpora's training "
                        "queries, then apply it frozen to every corpus in --datasets. The refiner "
                        "and decoder are corpus-agnostic (they consume centroids + page vectors), "
                        "so a codec fitted on A transfers to B by running B's own k-means through "
                        "it. This answers the reviewer question the in-corpus setting invites: "
                        "'do I need labelled queries for every corpus I want to compress?'. "
                        "Pass SEVERAL corpora to test whether cross-domain transfer improves with "
                        "source DIVERSITY -- batches are interleaved across them, and model "
                        "selection uses only the training corpora's holdouts, never the target.")
    p.add_argument("--train_on_split", default="test",
                   help="HF split for the --train_on corpora only. Set to 'train' to fit the codec "
                        "on vidore/colpali_train_set, which is the source Light-ColPali uses -- the "
                        "only way to compare the two zero-shot on MATCHED source data.")
    p.add_argument("--train_on_max_pages", type=int, default=0,
                   help="page cap for the --train_on corpora only (0 = use --max_pages). Kept "
                        "separate so capping a 118k-row training set does not invalidate the "
                        "evaluation caches, whose fingerprint includes max_pages.")
    p.add_argument("--model_name", default="vidore/colpali-v1.3")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--out_dir", default="./out")
    p.add_argument("--max_pages", type=int, default=0, help="0 = whole subset")
    p.add_argument("--encode_batch", type=int, default=8)

    # ---- split (identical for every method) ----------------------------------
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--eval_all_queries", action="store_true",
                   help="STANDARD-PROTOCOL EVAL: score every query of each --datasets corpus, as "
                        "the official benchmark does. Only valid with --train_on (zero-shot mode), "
                        "where no target query is ever used for fitting, so there is nothing to "
                        "hold out. The SOURCE corpora keep their --holdout_frac split for model "
                        "selection. Without --train_on this flag would evaluate on training "
                        "queries and is refused.")
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="fraction of the TRAINING queries actually used to fit the codec. The "
                        "holdout and the corpus are untouched, so points on the resulting curve are "
                        "directly comparable. Use for the data-scaling ablation: does the codec "
                        "saturate on a fraction of the labels?")

    # ---- budgets -------------------------------------------------------------
    p.add_argument("--codes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])

    # ---- model ---------------------------------------------------------------
    p.add_argument("--refiner", choices=["gated_tangent", "zero_init", "none"],
                   default="gated_tangent",
                   help="gated_tangent = bounded tangent-space update with a dispersion-conditioned "
                        "gate. zero_init = the simpler additive correction, renormalized. "
                        "none = normalized k-means passthrough (ablation).")
    p.add_argument("--decoder", choices=["anchored", "none"], default="anchored",
                   help="anchored = count-proportional slots with one exact anchor per cluster, so "
                        "the decoded set CONTAINS the code set. none = stage-1 only.")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dec_hidden", type=int, default=256)
    p.add_argument("--dec_layers", type=int, default=1,
                   help="DEPTH of the decoder's fusion stack, where --dec_hidden is its WIDTH. One "
                        "layer is LayerNorm -> expand to 2*hidden -> GELU -> contract -> GELU, and "
                        "1 is the published configuration, so a depth sweep is a controlled "
                        "comparison against the main table. Capacity here is free at rest: the "
                        "decoder is one network shared by every page and every corpus, so neither "
                        "width nor depth changes bytes per page. Only fitting time and the "
                        "stage-2 decode of the shortlist grow.")
    p.add_argument("--child_dims", type=int, default=1,
               help="number of coordinates naming a slot within its cluster. 1 reproduces the paper; larger values are an optional variant that is not reported.")
    p.add_argument("--detach_code", action="store_true",
               help="feed the decoder a detached code so decoder losses do not reach the refiner. Off in the paper; kept as an optional variant.")
    p.add_argument("--refiner_alpha_max", type=float, default=0.50)
    p.add_argument("--refiner_gate_bias", type=float, default=-4.0,
                   help="sigmoid(-4)=0.018, so the refiner starts ~exactly at normalized k-means")
    p.add_argument("--decoder_alpha_max", type=float, default=0.75)
    p.add_argument("--decoder_rank", type=int, default=0,
               help="rank of the decoder displacement subspace. 0 (unconstrained) reproduces the paper; r>0 is an optional variant that is not reported.")
    p.add_argument("--no_norm_channel", action="store_true",
                   help="ABLATION FOR THE PAPER'S CENTRAL CLAIM. Zeroes the stored magnitude r_j "
                        "before the decoder sees it, so the decoder knows WHERE each cluster sits "
                        "but not HOW WIDE it is. Since 1-r^2 is exactly the cluster's quantization "
                        "error, this tests whether the discarded k-means objective is actually the "
                        "useful signal. If the decoder does not degrade here, the paper's spine is "
                        "wrong and we need to know that before submission, not after.")
    p.add_argument("--no_count_channel", action="store_true",
                   help="companion ablation: zeroes the log-count feature, leaving the magnitude. "
                        "Separates 'how wide is this cluster' from 'how many patches are in it'.")

    # ---- training ------------------------------------------------------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min_epochs", type=int, default=30)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=5, help="evals without improvement (0 = off)")
    p.add_argument("--query_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    # ---- retrieval / rerank --------------------------------------------------
    p.add_argument("--topl", type=int, default=20, help="shortlist size rescored by the decoder")
    p.add_argument("--hard_negatives", type=int, default=7)
    p.add_argument("--hard_negative_pool", type=int, default=50)
    p.add_argument("--tail_policy", choices=["stage1", "strict"], default="stage1",
                   help="stage1 = non-candidates keep their stage-1 ordering (partial credit for "
                        "the tail, standard cascade accounting). strict = non-candidates are "
                        "dropped entirely (a shortlist miss scores 0). Report with 'stage1'; "
                        "'strict' is available because some baselines are evaluated that way.")

    # ---- losses (each weight is an ablation switch: set to 0 to drop the term) -
    p.add_argument("--w_cluster_set", type=float, default=0.50)
    p.add_argument("--w_support", type=float, default=0.10)
    p.add_argument("--w_gen_token", type=float, default=1.00)
    p.add_argument("--w_gen_listwise", type=float, default=1.00)
    p.add_argument("--w_overshoot", type=float, default=0.50)
    p.add_argument("--w_code_token", type=float, default=0.50)
    p.add_argument("--w_code_listwise", type=float, default=0.50)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--support_dirs", type=int, default=128)
    p.add_argument("--cluster_sample", type=int, default=48)

    # ---- bookkeeping ---------------------------------------------------------
    p.add_argument("--tag", default="main", help="names the results file")
    p.add_argument("--reuse", action="store_true", default=True,
                   help="skip a (dataset, k) whose checkpoint already exists")
    p.add_argument("--no_reuse", dest="reuse", action="store_false")
    return p


def loss_weights(args):
    return {
        "cluster_set": args.w_cluster_set,
        "support": args.w_support,
        "gen_token": args.w_gen_token,
        "gen_listwise": args.w_gen_listwise,
        "overshoot": args.w_overshoot,
        "code_token": args.w_code_token,
        "code_listwise": args.w_code_listwise,
    }


def ensure_dirs(args):
    for d in [args.cache_dir, args.out_dir,
              os.path.join(args.out_dir, "checkpoints"),
              os.path.join(args.out_dir, "results")]:
        os.makedirs(d, exist_ok=True)
    return args
