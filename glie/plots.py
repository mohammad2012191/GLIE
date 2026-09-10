"""Publication figures, read straight from the result CSVs.

Produces (PDF + PNG):
  fig_storage_curve   nDCG@5 vs storage bytes/page
  fig_margin_<bench>  margin over the best training-free baseline vs k, one panel per benchmark
  fig_headroom        where the remaining error lives at each budget
  fig_readout_lift    what the generative read-out adds over the stored code, per benchmark
  fig_lcp_bars        GLIE vs Light-ColPali at a matched training budget

Usage
-----
    python -m glie.plots --style paper           # column-width, for LaTeX
    python -m glie.plots --style large           # large type, for slides

The "best training-free baseline" is computed EXACTLY as run_main does it: the strongest of
{norm_kmeans, cluster_merge, token_pool} chosen per (dataset, budget), then macro-averaged over
datasets. Any other aggregation would flatter us.

Palette is the semantic one from the paper's pipeline figures, so diagrams and plots read as one
artifact:
    blue     original / teacher vectors  -> uncompressed ceiling
    magenta  projected vectors           -> the stored code, stage 1
    green    regenerated vectors         -> the decoder, full system
    slate    baselines
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
C = {"ceiling": "#3B6FD4",      # blue: original / teacher vectors, as in Figures 1-2
     "baseline": "#94A3B8",     # slate: prior training-free methods
     "stage1": "#AE2D9C",       # magenta: the projected vectors that are stored
     "decoder": "#128A5E",      # green: the regenerated vectors
     "lcp": "#8B5CF6", "shade": "#FEF3C7", "rule": "#EF4444",
     "tint_dec": "#A8DFC8",     # light green: the read-out's own contribution, in the margin plot
     "headroom_dec": "#F2B950",  # amber: headroom a BETTER decoder would reach (unachieved)
     "tint_ceil": "#C7D7F5"}    # light blue: headroom attributable to the shortlist


def bt_x(ax):
    """x in data coords, y in axes fraction."""
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


def style(mode):
    if mode == "large":
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
# The storage / accuracy curve
# ---------------------------------------------------------------------------
def fig_storage_curve(series, ceiling, bytes_per_k, n_patches, s, out_dir, dim=128,
                      whitespace_k=16, lcp_floor_k=16, ymin=None):
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
    lo = ymin if ymin is not None else min(series["best_tf"].min(), 0.42) - 0.03
    ax.set_ylim(lo, ceiling + 0.045)
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
def fig_margin(margins, s, out_dir):
    """`margins` is either one benchmark's dict (keys glie_decoder / glie_stage1) or a dict of
    those keyed by benchmark name. With several benchmarks, colour = benchmark and line style =
    stage: the full system solid with filled markers, stage 1 dashed with hollow markers."""
    if "glie_decoder" in margins:
        margins = {"ViDoRe v1": margins}

    # One panel per benchmark, on SHARED y-limits: the point of the pair is that v2's margins do
    # not decay while v1's do, and separate autoscaled axes would hide exactly that.
    allv = [v for m in margins.values() for key in ("glie_decoder", "glie_stage1")
            for v in m[key].values]
    lo, hi = min(0.0, min(allv)), max(allv)
    pad = (hi - lo) * 0.10
    slug = {"ViDoRe v1": "v1", "ViDoRe v2": "v2"}

    for bench, m in margins.items():
        fig, ax = plt.subplots(figsize=s["figsize"])
        kb = np.array(sorted(m["glie_decoder"].index))
        yd = np.array([m["glie_decoder"][k] for k in kb])
        ys = np.array([m["glie_stage1"][k] for k in kb])

        ax.axhline(0, color="#CBD5E1", lw=1.4, zorder=1)

        # the band between the two curves IS the generative read-out's contribution
        ax.fill_between(kb, ys, yd, color=C["tint_dec"], alpha=0.9, zorder=2,
                        label="generative read-out")
        ax.plot(kb, ys, "--o", color=C["stage1"], lw=s["lw"] * 0.9, ms=s["ms"],
                markerfacecolor="white", markeredgewidth=1.8, label="stage 1 (stored code)", zorder=4)
        ax.plot(kb, yd, "-o", color=C["decoder"], lw=s["lw"], ms=s["ms"],
                label="full system", zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(kb); ax.set_xticklabels([str(k) for k in kb])
        apply(ax, s, "vectors stored per page", "nDCG@5 gain over best baseline", title=bench)
        ax.set_xlim(kb.min() * 0.8, kb.max() * 1.25)
        ax.set_ylim(lo - pad, hi + pad)
        ax.legend(fontsize=s["legend"], loc="upper right", frameon=True, framealpha=0.95)
        save(fig, out_dir, "fig_margin_" + slug.get(bench, bench.lower().replace(" ", "_")))


# ---------------------------------------------------------------------------
# Headroom decomposition: where the remaining error lives at each budget
# ---------------------------------------------------------------------------
def fig_headroom(series, ceiling, s, out_dir, ymin=0.5):
    """Stacked bars per budget. From the bottom: the stored code's score, what the generative
    read-out adds, what a PERFECT decoder on the same shortlist would still add (oracle minus
    read-out = decode fidelity), and what a larger shortlist would add (ceiling minus oracle =
    shortlist recall). Solid tints of the paper palette, no hatching, so it reads like the
    line figures. The y-axis starts at `ymin`; the three upper blocks are absolute differences
    and are unaffected by where the axis starts."""
    ks = sorted(series["glie_decoder"].index)
    s1 = np.array([series["glie_stage1"][k] for k in ks])
    dec = np.array([series["glie_decoder"][k] for k in ks])
    orc = np.array([series["glie_oracle"][k] for k in ks])
    fig, ax = plt.subplots(figsize=s["figsize"])
    x = np.arange(len(ks)); w = 0.64
    # achieved blocks keep the semantic palette; the two headroom blocks are deliberately
    # OFF that palette (amber, pale blue) so they never read as more of the same thing
    ax.bar(x, s1 - ymin, w, bottom=ymin, color=C["stage1"], label="stage 1", zorder=3)
    ax.bar(x, dec - s1, w, bottom=s1, color=C["decoder"], label="+ read-out", zorder=3)
    ax.bar(x, orc - dec, w, bottom=dec, color=C["headroom_dec"],
           label="decode headroom", zorder=3)
    ax.bar(x, ceiling - orc, w, bottom=orc, color=C["tint_ceil"],
           label="shortlist headroom", zorder=3)
    ax.axhline(ceiling, color=C["ceiling"], ls="--", lw=s["lw"] * 0.7, zorder=2)
    ax.text(x[0] - w / 2, ceiling + 0.006, f"uncompressed  {ceiling:.3f}", ha="left", va="bottom",
            fontsize=s["ann"] * 0.9, color=C["ceiling"])
    for i in range(len(ks)):                      # size of the decode-fidelity block
        gap = orc[i] - dec[i]
        if gap >= 0.045:                          # label fits inside the block
            ax.text(x[i], dec[i] + gap / 2, f"{gap:.3f}", ha="center", va="center",
                    fontsize=s["ann"] * 1.05, color="#7A4A00", fontweight="bold")
        else:                                     # too thin: label above the ceiling line
            ax.text(x[i], ceiling + 0.006, f"{gap:.3f}", ha="center", va="bottom",
                    fontsize=s["ann"] * 1.05, color="#7A4A00", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
    apply(ax, s, "vectors stored per page", "nDCG@5")
    ax.set_ylim(ymin, ceiling + 0.05)
    # one row of short labels: a two-column legend is wider than the axes and pads the crop
    ax.legend(fontsize=s["legend"] * 0.92, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False, columnspacing=1.4, handlelength=1.4, handletextpad=0.5)
    save(fig, out_dir, "fig_headroom")


# ---------------------------------------------------------------------------
# Read-out lift over stage 1, per benchmark: the component that keeps paying where headroom survives
# ---------------------------------------------------------------------------
def fig_readout_lift(series_by_bench, s, out_dir):
    """decoder - stage1 at each budget, one line per benchmark. On v1 it peaks at aggressive
    budgets and decays as the code saturates; on v2, which saturates nowhere, it does not."""
    fig, ax = plt.subplots(figsize=s["figsize"])
    styles = [(C["decoder"], "o", "-"), ("#0F766E", "s", "--"), (C["lcp"], "D", ":")]
    for (name, series), (col, mk, ls) in zip(series_by_bench.items(), styles):
        ks = np.array(sorted(series["glie_decoder"].index))
        y = np.array([series["glie_decoder"][k] - series["glie_stage1"][k] for k in ks])
        ax.plot(ks, y, ls, marker=mk, color=col, lw=s["lw"], ms=s["ms"], label=name, zorder=5)
    ax.axhline(0, color="#CBD5E1", lw=1.4, zorder=1)
    ax.set_xscale("log", base=2)
    ks_all = sorted({k for sr in series_by_bench.values() for k in sr["glie_decoder"].index})
    ax.set_xticks(ks_all); ax.set_xticklabels([str(k) for k in ks_all])
    apply(ax, s, "vectors stored per page", "nDCG@5 gain of the read-out over stage 1")
    ax.set_xlim(min(ks_all) * 0.8, max(ks_all) * 1.25)
    ax.legend(fontsize=s["legend"], loc="upper right", frameon=True, framealpha=0.95)
    save(fig, out_dir, "fig_readout_lift")


# ---------------------------------------------------------------------------
# GLIE vs Light-ColPali at matched training budget
# ---------------------------------------------------------------------------
def fig_lcp_bars(lcp_path, glie_csv, s, out_dir, metric="ndcg@5", setting=""):
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
    p.add_argument("--results", default="./out/results")
    p.add_argument("--tag", default="std_main",
                   help="standard-protocol main run; the old in-corpus default was vidore_full")
    p.add_argument("--lcp_tag", default="lcp_matched_fixed_2")
    p.add_argument("--lcp_glie_tag", default="",
                   help="GLIE run to put beside Light-ColPali in the bar chart. Defaults to --tag, "
                        "but Light-ColPali is ZERO-SHOT on the eval corpora, so pairing it with an "
                        "in-corpus GLIE run is not a fair comparison -- point this at the transfer "
                        "run (e.g. glie_transfer_cptrain) and say so in the caption.")
    p.add_argument("--lcp_setting", default="both zero-shot",
                   help="short setting label printed in the bar-chart title")
    p.add_argument("--out", default="figures")
    p.add_argument("--style", choices=["paper", "large"], default="paper")
    p.add_argument("--metric", default="ndcg@5")
    p.add_argument("--ymin", type=float, default=None,
                   help="lower y-limit of the storage curve (default: auto, below the lowest baseline)")
    p.add_argument("--headroom_ymin", type=float, default=0.5,
                   help="lower y-limit of the headroom bars")
    p.add_argument("--v2_tag", default="v2_std",
                   help="ViDoRe v2 results tag for the read-out lift figure ('' to skip)")
    p.add_argument("--floor_k", type=int, default=16,
                   help="published post-hoc floor (no prior method reports below ~16 vectors), drawn on the storage curve")
    a = p.parse_args()

    s = style(a.style)
    out = os.path.join(a.out, a.style)
    csv = os.path.join(a.results, f"{a.tag}__main_table.csv")
    print(f"reading {csv}")
    series, margins, ceiling, bytes_per_k, n_patches = load_table(csv, a.metric)
    print(f"  ceiling={ceiling:.4f}  budgets={sorted(series['best_tf'].index)}  P={n_patches}")

    fig_storage_curve(series, ceiling, bytes_per_k, n_patches, s, out,
                      whitespace_k=a.floor_k, lcp_floor_k=a.floor_k, ymin=a.ymin)
    fig_headroom(series, ceiling, s, out, ymin=a.headroom_ymin)

    benches, margins_by = {"ViDoRe v1": series}, {"ViDoRe v1": margins}
    v2csv = os.path.join(a.results, f"{a.v2_tag}__main_table.csv")
    if a.v2_tag and os.path.exists(v2csv):
        s2, m2, *_ = load_table(v2csv, a.metric)
        benches["ViDoRe v2"], margins_by["ViDoRe v2"] = s2, m2
    fig_margin(margins_by, s, out)
    fig_readout_lift(benches, s, out)

    lcp = os.path.join(a.results, f"{a.lcp_tag}__results.json")
    glie_for_bars = os.path.join(a.results,
                                 f"{a.lcp_glie_tag or a.tag}__main_table.csv")
    if os.path.exists(lcp) and os.path.exists(glie_for_bars):
        if not a.lcp_glie_tag:
            print("  [lcp] WARNING: pairing Light-ColPali with --tag. If that run is in-corpus the "
                  "comparison is unfair -- pass --lcp_glie_tag <transfer run>.")
        fig_lcp_bars(lcp, glie_for_bars, s, out, a.metric, a.lcp_setting)
    else:
        print(f"  [lcp] missing {lcp} or {glie_for_bars}, skipping the baseline bars")

    print(f"\nfigures in {out}/")


if __name__ == "__main__":
    main()
