"""Publication figures, read straight from the result CSVs.

Produces (PDF + PNG):
  fig_storage_curve   nDCG@5 vs storage bytes/page  -> poster element 2 / paper Fig 2
  fig_margin          margin over the best baseline vs k, with the covering-demand line
                      -> poster element 5 / paper Fig 4
  fig_lcp_bars        GLIE vs Light-ColPali at matched budget -> poster element 6

Usage
-----
    python -m glie.plots --style poster          # big type, for A0
    python -m glie.plots --style paper           # column-width, for LaTeX

The "best training-free baseline" is computed EXACTLY as run_main does it: the strongest of
{norm_kmeans, cluster_merge, token_pool} chosen per (dataset, budget), then macro-averaged over
datasets. Any other aggregation would flatter us.

Palette is the semantic one from Figure 1, so the poster and the paper read as one artifact:
    blue   original patches / ceiling
    orange the stored code  -> stage 1
    teal   regenerated      -> decoder
    slate  baselines
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TF_METHODS = ["norm_kmeans", "cluster_merge", "token_pool"]
C = {"ceiling": "#64748B", "baseline": "#94A3B8", "stage1": "#F97316",
     "decoder": "#10B981", "lcp": "#8B5CF6", "shade": "#FEF3C7", "rule": "#EF4444"}


def bt_x(ax):
    """x in data coords, y in axes fraction."""
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


def style(mode):
    if mode == "poster":
        return {"figsize": (11, 7.5), "title": 30, "label": 26, "tick": 22, "legend": 22,
                "lw": 4.5, "ms": 14, "ann": 20}
    return {"figsize": (7.0, 4.6), "title": 15, "label": 14, "tick": 12, "legend": 12,
            "lw": 2.4, "ms": 7, "ann": 11}


def apply(ax, s, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=s["label"])
    ax.set_ylabel(ylabel, fontsize=s["label"])
    if title:
        ax.set_title(title, fontsize=s["title"], pad=12)
    ax.tick_params(labelsize=s["tick"])
    ax.grid(alpha=0.25, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"  wrote {os.path.join(out_dir, name)}.{{pdf,png}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation (identical to run_main's margin computation)
# ---------------------------------------------------------------------------
def load_table(path, metric="ndcg@5"):
    df = pd.read_csv(path)
    body = df[df.method != "ceiling"]
    ceiling = float(df[df.method == "ceiling"][metric].mean())

    best_tf_pd = body[body.method.isin(TF_METHODS)].groupby(["dataset", "k"])[metric].max()
    series = {"best_tf": best_tf_pd.groupby("k").mean()}
    for m in ["glie_stage1", "glie_decoder", "glie_oracle"] + TF_METHODS + ["raw_kmeans"]:
        sub = body[body.method == m]
        if len(sub):
            series[m] = sub.groupby("k")[metric].mean()

    margins = {}
    for m in ["glie_stage1", "glie_decoder"]:
        g = body[body.method == m].set_index(["dataset", "k"])[metric]
        common = g.index.intersection(best_tf_pd.index)
        margins[m] = (g.loc[common] - best_tf_pd.loc[common]).groupby("k").mean()

    bytes_per_k = body[body.method == "glie_decoder"].groupby("k")["bytes_per_page"].first()
    return series, margins, ceiling, bytes_per_k, int(df[df.method == "ceiling"]["k"].max())


# ---------------------------------------------------------------------------
# Figure 1 of the poster: the storage / accuracy curve
# ---------------------------------------------------------------------------
def fig_storage_curve(series, ceiling, bytes_per_k, n_patches, s, out_dir, dim=128,
                      whitespace_k=16, lcp_floor_k=16):
    from matplotlib.transforms import blended_transform_factory
    fig, ax = plt.subplots(figsize=s["figsize"])
    ks = np.array(sorted(series["best_tf"].index))
    x = np.array([bytes_per_k.get(k, k * dim * 2) for k in ks], dtype=float)
    # x in data coords, y in axes fraction: keeps annotations off the curves whatever the y-range
    bt = blended_transform_factory(ax.transData, ax.transAxes)

    # the regime no published post-hoc method reaches
    ax.axvspan(x.min() * 0.72, whitespace_k * dim * 2, color=C["shade"], alpha=0.55, zorder=0)
    ax.text(0.028, 0.82, "no published post-hoc\nmethod operates here",
            transform=ax.transAxes, fontsize=s["ann"], color="#92400E",
            va="top", ha="left", zorder=3)
    ax.axvline(lcp_floor_k * dim * 2, color=C["rule"], ls=":", lw=s["lw"] * 0.55, zorder=1)
    ax.text(lcp_floor_k * dim * 2 * 1.06, 0.975, "$\\leftarrow$ baseline floor",
            transform=bt, fontsize=s["ann"] * 0.85, color=C["rule"],
            va="top", ha="left", zorder=3)

    # ceiling labelled on the LEFT, above its own line: the top-right belongs to the floor marker
    ax.axhline(ceiling, color=C["ceiling"], ls="--", lw=s["lw"] * 0.7, zorder=2)
    ax.text(x.min() * 0.80, ceiling + 0.006,
            f"uncompressed  ({n_patches} vec, {n_patches * dim * 2 / 1024:.0f} KB)",
            fontsize=s["ann"] * 0.9, color=C["ceiling"], ha="left", va="bottom")

    for key, lab, col, lw_mult, z in [
        ("best_tf", "best training-free baseline", C["baseline"], 0.85, 4),
        ("glie_stage1", "GLIE  (stage 1, single-stage)", C["stage1"], 1.0, 5),
        ("glie_decoder", "GLIE  (+ generative rerank)", C["decoder"], 1.25, 6),
    ]:
        if key not in series:
            continue
        y = np.array([series[key][k] for k in ks])
        ax.plot(x, y, "-o", color=col, lw=s["lw"] * lw_mult, ms=s["ms"], label=lab, zorder=z)

    # % of ceiling, labelled just above the decoder curve so nothing crosses a line
    for k_ann in (ks[1], ks[len(ks) // 2]):
        if k_ann in series["glie_decoder"].index:
            v = series["glie_decoder"][k_ann]
            xb = bytes_per_k.get(k_ann, k_ann * dim * 2)
            ax.annotate(f"{100 * v / ceiling:.0f}%", xy=(xb, v), xytext=(xb, v + 0.022),
                        fontsize=s["ann"] * 1.05, color=C["decoder"], fontweight="bold",
                        ha="center", va="bottom")

    ax.set_xscale("log")
    apply(ax, s, "storage per page (bytes, log scale)", "nDCG@5")
    ax.set_ylim(min(series["best_tf"].min(), 0.42) - 0.03, ceiling + 0.045)
    ax.set_xlim(x.min() * 0.72, x.max() * 1.55)

    sec = ax.secondary_xaxis("top")
    sec.set_xticks(x); sec.set_xticklabels([str(k) for k in ks])
    sec.set_xlabel("vectors stored per page", fontsize=s["label"] * 0.92)
    sec.tick_params(labelsize=s["tick"] * 0.9)

    ax.legend(fontsize=s["legend"], loc="lower right", frameon=True, framealpha=0.95)
    save(fig, out_dir, "fig_storage_curve")


# ---------------------------------------------------------------------------
# The margin figure: where the value concentrates
# ---------------------------------------------------------------------------
def fig_margin(margins, s, out_dir, covering_k=16):
    fig, ax = plt.subplots(figsize=s["figsize"])
    ks = np.array(sorted(margins["glie_decoder"].index))

    ax.axvspan(ks.min() * 0.8, covering_k, color=C["shade"], alpha=0.5, zorder=0)
    ax.axvline(covering_k, color=C["rule"], ls="--", lw=s["lw"] * 0.6, zorder=1)
    ax.text(covering_k * 1.1, max(margins["glie_decoder"]) * 0.92,
            "predicted covering\ndemand  (Lemma 1)",
            fontsize=s["ann"], color=C["rule"], va="top")
    ax.axhline(0, color="#CBD5E1", lw=1.4, zorder=1)

    for key, lab, col in [("glie_decoder", "GLIE (+ rerank)", C["decoder"]),
                          ("glie_stage1", "GLIE (stage 1)", C["stage1"])]:
        y = np.array([margins[key][k] for k in ks])
        ax.plot(ks, y, "-o", color=col, lw=s["lw"], ms=s["ms"], label=lab, zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    apply(ax, s, "vectors stored per page", "nDCG@5 gain over best baseline")
    ax.set_xlim(ks.min() * 0.8, ks.max() * 1.25)
    # lower left: the curves live in the upper band, and the Lemma-1 note owns the upper right
    ax.legend(fontsize=s["legend"], loc="lower left", frameon=True, framealpha=0.95)
    save(fig, out_dir, "fig_margin")


# ---------------------------------------------------------------------------
# GLIE vs Light-ColPali at matched training budget
# ---------------------------------------------------------------------------
def fig_lcp_bars(lcp_path, glie_csv, s, out_dir, metric="ndcg@5", setting="", covering_k=16):
    """GLIE vs Light-ColPali. `glie_csv` should be the run whose TRAINING SETTING matches the
    baseline's. Light-ColPali is fitted on colpali_train_set and applied zero-shot, so pairing it
    with an IN-CORPUS GLIE run gives GLIE target-corpus supervision the baseline never had. Point
    this at the transfer run (--lcp_glie_tag) and say so in the caption."""
    rows = json.load(open(lcp_path))
    lcp = pd.DataFrame(rows)
    if metric not in lcp.columns:
        print("  [lcp] metric column missing, skipping"); return
    subsets = sorted(lcp.dataset.unique())

    g = pd.read_csv(glie_csv)
    g = g[g.dataset.isin(subsets)]                    # SAME subsets, or the bars are not comparable
    if not len(g):
        print(f"  [lcp] no overlapping subsets between the two runs, skipping"); return

    lcp_m = lcp.groupby("k")[metric].mean()
    glie_m = g[g.method == "glie_decoder"].groupby("k")[metric].mean()
    nk_m = g[g.method == "norm_kmeans"].groupby("k")[metric].mean()
    ks = [k for k in sorted(glie_m.index) if k in lcp_m.index]

    fig, ax = plt.subplots(figsize=s["figsize"])
    idx = np.arange(len(ks))

    # Markers, not bars. Every value sits in a narrow band (~0.45-0.70) and the differences that
    # matter are ~0.005-0.05, so a zero-based bar chart makes them invisible while a TRUNCATED bar
    # chart exaggerates them -- a distortion reviewers rightly flag. Markers make a zoomed y-range
    # conventional and honest.
    for vals, lab, col, mk in [
        (nk_m, "normalized k-means (free)", C["baseline"], "o"),
        (lcp_m, "Light-ColPali (fine-tuned)", C["lcp"], "s"),
        (glie_m, "GLIE (frozen backbone)", C["decoder"], "D"),
    ]:
        ax.plot(idx, [vals[k] for k in ks], mk + "-", color=col, label=lab,
                lw=s["lw"] * 0.9, ms=s["ms"] * 1.15, markeredgecolor="white", markeredgewidth=1.6)

    # the covering demand: left of it learned compression pays, right of it free clustering already
    # is near-optimal. Without this the reader sees "their method stops working" instead of
    # "the theory predicted exactly this".
    if covering_k in ks:
        cut = ks.index(covering_k) - 0.5
        ax.axvspan(-0.5, cut, color=C["shade"], alpha=0.45, zorder=0)
        ax.axvline(cut, color=C["rule"], ls="--", lw=s["lw"] * 0.55, zorder=1)
        ax.text(cut - 0.12, 0.035, "learned compression pays", transform=bt_x(ax),
                fontsize=s["ann"] * 0.9, color="#92400E", ha="right", va="bottom")
        ax.text(cut + 0.12, 0.035, "free clustering already near-optimal", transform=bt_x(ax),
                fontsize=s["ann"] * 0.9, color=C["ceiling"], ha="left", va="bottom")

    ax.set_xticks(idx); ax.set_xticklabels([f"k={k}" for k in ks])
    title = f"matched budget{', ' + setting if setting else ''}, {len(subsets)} subsets"
    apply(ax, s, "vectors stored per page", "nDCG@5", title)
    lo = min(nk_m.min(), lcp_m.min(), glie_m.min())
    hi = max(nk_m.max(), lcp_m.max(), glie_m.max())
    ax.set_ylim(lo - (hi - lo) * 0.28, hi + (hi - lo) * 0.34)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.legend(fontsize=s["legend"], loc="upper left", frameon=True, framealpha=0.95)
    save(fig, out_dir, "fig_lcp_bars")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/ibex/user/hamidme/glie_out/results")
    p.add_argument("--tag", default="vidore_full")
    p.add_argument("--lcp_tag", default="lcp_matched_fixed_2")
    p.add_argument("--lcp_glie_tag", default="",
                   help="GLIE run to put beside Light-ColPali in the bar chart. Defaults to --tag, "
                        "but Light-ColPali is ZERO-SHOT on the eval corpora, so pairing it with an "
                        "in-corpus GLIE run is not a fair comparison -- point this at the transfer "
                        "run (e.g. glie_transfer_cptrain) and say so in the caption.")
    p.add_argument("--lcp_setting", default="both zero-shot",
                   help="short setting label printed in the bar-chart title")
    p.add_argument("--out", default="figures")
    p.add_argument("--style", choices=["paper", "poster"], default="poster")
    p.add_argument("--metric", default="ndcg@5")
    p.add_argument("--covering_k", type=int, default=16,
                   help="budget at which Lemma 1 predicts the covering demand is crossed")
    a = p.parse_args()

    s = style(a.style)
    out = os.path.join(a.out, a.style)
    csv = os.path.join(a.results, f"{a.tag}__main_table.csv")
    print(f"reading {csv}")
    series, margins, ceiling, bytes_per_k, n_patches = load_table(csv, a.metric)
    print(f"  ceiling={ceiling:.4f}  budgets={sorted(series['best_tf'].index)}  P={n_patches}")

    fig_storage_curve(series, ceiling, bytes_per_k, n_patches, s, out,
                      whitespace_k=a.covering_k, lcp_floor_k=a.covering_k)
    fig_margin(margins, s, out, covering_k=a.covering_k)

    lcp = os.path.join(a.results, f"{a.lcp_tag}__results.json")
    glie_for_bars = os.path.join(a.results,
                                 f"{a.lcp_glie_tag or a.tag}__main_table.csv")
    if os.path.exists(lcp) and os.path.exists(glie_for_bars):
        if not a.lcp_glie_tag:
            print("  [lcp] WARNING: pairing Light-ColPali with --tag. If that run is in-corpus the "
                  "comparison is unfair -- pass --lcp_glie_tag <transfer run>.")
        fig_lcp_bars(lcp, glie_for_bars, s, out, a.metric, a.lcp_setting, a.covering_k)
    else:
        print(f"  [lcp] missing {lcp} or {glie_for_bars}, skipping the baseline bars")

    print(f"\nfigures in {out}/")


if __name__ == "__main__":
    main()
