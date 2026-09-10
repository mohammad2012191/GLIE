"""Per-page manifold statistics on all ten ViDoRe v1 subsets.

Reads the same caches the main results use, so the geometry is measured on exactly the
embeddings that were retrieved against. Padding patches are excluded via pmask.

Three statistics per page, aggregated per subset and pooled:
  1. TwoNN intrinsic dimension (Facco et al. 2017)
  2. participation ratio of the covariance spectrum (effective linear dimension)
  3. k-means inertia decay exponent: fit inertia ~ k^{-alpha} over k in KS; d_eff = 2/alpha

Run from the parent of glie/:
    python -m glie.geometry_study --cache_dir ./cache --out ./out/results/geometry.json
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

SLUGS = [
    "arxivqa_test_subsampled", "docvqa_test_subsampled", "infovqa_test_subsampled",
    "tabfquad_test_subsampled", "tatdqa_test", "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test", "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test", "syntheticDocQA_healthcare_industry_test",
]
KS = [2, 4, 8, 16, 32, 64, 128]


def twonn(X):
    """Intrinsic dimension, Facco et al. 2017. X: [n, D] float64."""
    dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    ok = dist[:, 1] > 0
    mu = dist[ok, 2] / dist[ok, 1]
    mu = np.sort(mu[mu > 1])
    if len(mu) < 10:
        return np.nan
    F = np.arange(1, len(mu) + 1) / len(mu)
    x, y = np.log(mu[:-1]), -np.log(1 - F[:-1])
    return float((x * y).sum() / (x * x).sum())


def participation_ratio(X):
    lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def inertia_alpha(X, ks=KS, seed=0):
    """Fit inertia ~ k^{-alpha} by least squares in log-log; also return the per-k inertias so the
    'no elbow / smooth continuum' claim can be checked from the saved output, not re-run."""
    ks = [k for k in ks if k < len(X)]
    inert = [KMeans(k, n_init=3, random_state=seed).fit(X).inertia_ for k in ks]
    lx, ly = np.log(np.array(ks, float)), np.log(np.array(inert, float))
    alpha = -np.polyfit(lx, ly, 1)[0]
    return float(alpha), {int(k): float(v) for k, v in zip(ks, inert)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha_pages", type=int, default=100,
                    help="pages per subset for the k-means slope (the expensive statistic); "
                         "TwoNN and PR always use every page. 0 = all pages for alpha too.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    report, pooled = {}, {"twonn": [], "pr": [], "alpha": []}
    for slug in SLUGS:
        page = torch.load(os.path.join(args.cache_dir, f"{slug}__page.pt")).numpy()
        pmask = torch.load(os.path.join(args.cache_dir, f"{slug}__pmask.pt")).numpy()
        n = page.shape[0]

        ids, prs = [], []
        for i in range(n):
            X = page[i][pmask[i]].astype(np.float64)
            ids.append(twonn(X))
            prs.append(participation_ratio(X))

        sub = rng.permutation(n)[: args.alpha_pages] if args.alpha_pages else np.arange(n)
        alphas, curves = [], []
        for i in sub:
            a, c = inertia_alpha(page[i][pmask[i]].astype(np.float64), seed=args.seed)
            alphas.append(a)
            curves.append(c)

        ids, prs, alphas = map(np.asarray, (ids, prs, alphas))
        report[slug] = {
            "pages": int(n),
            "twonn_median": float(np.nanmedian(ids)), "twonn_iqr":
                [float(np.nanpercentile(ids, 25)), float(np.nanpercentile(ids, 75))],
            "pr_median": float(np.median(prs)),
            "alpha_median": float(np.median(alphas)), "alpha_pages": int(len(sub)),
            "d_eff_from_alpha": float(2.0 / np.median(alphas)),
            "mean_inertia_curve": {k: float(np.mean([c[k] for c in curves if k in c]))
                                   for k in KS if any(k in c for c in curves)},
        }
        pooled["twonn"].extend(ids[~np.isnan(ids)].tolist())
        pooled["pr"].extend(prs.tolist())
        pooled["alpha"].extend(alphas.tolist())
        r = report[slug]
        print(f"{slug:<45} pages={n:<5} TwoNN={r['twonn_median']:.2f} "
              f"PR={r['pr_median']:.2f} alpha={r['alpha_median']:.3f} "
              f"(d_eff={r['d_eff_from_alpha']:.2f})")

    report["POOLED"] = {
        "twonn_median": float(np.median(pooled["twonn"])),
        "pr_median": float(np.median(pooled["pr"])),
        "alpha_median": float(np.median(pooled["alpha"])),
        "d_eff_from_alpha": float(2.0 / np.median(pooled["alpha"])),
        "twonn_subset_range": [min(report[s]["twonn_median"] for s in SLUGS),
                               max(report[s]["twonn_median"] for s in SLUGS)],
    }
    p = report["POOLED"]
    print(f"\nPOOLED over {len(pooled['twonn'])} pages: TwoNN={p['twonn_median']:.2f}  "
          f"PR={p['pr_median']:.2f}  alpha={p['alpha_median']:.3f}  "
          f"d_eff={p['d_eff_from_alpha']:.2f}")
    print(f"TwoNN per-subset median range: {p['twonn_subset_range'][0]:.2f}"
          f" -- {p['twonn_subset_range'][1]:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
