"""Null models for the intrinsic-dimension claim.

Section 4.1 reports TwoNN ~ 5 against a participation ratio ~ 25 and reads the gap as a curved
low-dimensional manifold inside a wider linear span. That reading has never been checked against
what the same estimator returns on data with NO such structure at the same sample size. TwoNN is
biased downward in high ambient dimension with ~1000 points, so a small number is only evidence
if noise gives a larger one.

Three controls, each built per page with that page's own N and D so nothing else differs:

    uniform    Gaussian in D, renormalized to the sphere. No structure at all. This is the
               estimator's ceiling at this N and D; it will NOT be 128, and the real value is
               only meaningful relative to it.
    gaussian   N(mu, Sigma) fitted to the page's real tokens, renormalized. Same covariance
               spectrum as the real page, so the same participation ratio, but no curvature.
               THE decisive control: a Gaussian's intrinsic dimension equals its linear
               dimension, so if this gives TwoNN ~ PR ~ 25 while the real page gives 5, the
               gap is curvature and the manifold claim stands. If this also gives ~5, the
               claim is an artifact.
    shuffle    Each coordinate permuted independently across tokens. Marginals preserved,
               all joint structure destroyed.

CPU only, no training.

    python -m glie.geometry_control --cache_dir ./cache --out ./out/results/geometry_control.json
"""
import argparse
import json
import os

import numpy as np
import torch

from .geometry_study import SLUGS, participation_ratio, twonn


def _sphere(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def make_controls(X, rng):
    """X: [n, D] real unit vectors of one page -> dict of three [n, D] null samples."""
    n, D = X.shape
    uniform = _sphere(rng.standard_normal((n, D)))
    mu = X.mean(0)
    # Sample from the fitted Gaussian via the SVD of the centered data, which is exact and
    # avoids forming a rank-deficient D x D covariance when n < D.
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = rng.standard_normal((n, len(s)))
    gaussian = _sphere(mu + (Z * (s / np.sqrt(max(n - 1, 1)))) @ Vt)
    shuffle = X.copy()
    for j in range(D):
        shuffle[:, j] = rng.permutation(shuffle[:, j])
    shuffle = _sphere(shuffle)
    return {"uniform": uniform, "gaussian": gaussian, "shuffle": shuffle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slugs", nargs="+", default=SLUGS)
    ap.add_argument("--pages", type=int, default=200,
                    help="pages per subset (0 = all). 200 is enough: TwoNN on a page is one "
                         "number and the medians stabilize well before that.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    NAMES = ["real", "uniform", "gaussian", "shuffle"]
    report, pooled = {}, {f"{k}_{m}": [] for k in NAMES for m in ("twonn", "pr")}
    for slug in args.slugs:
        page = torch.load(os.path.join(args.cache_dir, f"{slug}__page.pt")).float().numpy()
        pmask = torch.load(os.path.join(args.cache_dir, f"{slug}__pmask.pt")).numpy()
        idx = rng.permutation(page.shape[0])[: args.pages] if args.pages else np.arange(page.shape[0])

        acc = {f"{k}_{m}": [] for k in NAMES for m in ("twonn", "pr")}
        for i in idx:
            X = page[i][pmask[i]].astype(np.float64)
            if len(X) < 20:
                continue
            samples = {"real": X, **make_controls(X, rng)}
            for k, S in samples.items():
                acc[f"{k}_twonn"].append(twonn(S))
                acc[f"{k}_pr"].append(participation_ratio(S))
        rec = {key: float(np.nanmedian(v)) for key, v in acc.items()}
        rec["pages"] = int(len(idx))
        rec["n_tokens_median"] = float(np.median([int(pmask[i].sum()) for i in idx]))
        report[slug] = rec
        for key, v in acc.items():
            pooled[key].extend([x for x in v if not np.isnan(x)])
        print(f"{slug:<45} n={rec['n_tokens_median']:.0f}  "
              + "  ".join(f"{k}: d={rec[f'{k}_twonn']:5.2f} PR={rec[f'{k}_pr']:5.1f}"
                          for k in NAMES))

    P = {key: float(np.median(v)) for key, v in pooled.items()}
    report["POOLED"] = P
    print("\n" + "=" * 100)
    print(f"{'':<10}{'TwoNN':>8}{'PR':>8}   reading")
    print(f"{'real':<10}{P['real_twonn']:>8.2f}{P['real_pr']:>8.1f}   the claim: d << PR")
    print(f"{'uniform':<10}{P['uniform_twonn']:>8.2f}{P['uniform_pr']:>8.1f}   "
          f"estimator ceiling at this n, D; real must sit well below this")
    print(f"{'gaussian':<10}{P['gaussian_twonn']:>8.2f}{P['gaussian_pr']:>8.1f}   "
          f"same spectrum, no curvature; d should be ~PR here and << PR for real")
    print(f"{'shuffle':<10}{P['shuffle_twonn']:>8.2f}{P['shuffle_pr']:>8.1f}   "
          f"marginals only")
    gap_real = P["real_pr"] - P["real_twonn"]
    gap_gauss = P["gaussian_pr"] - P["gaussian_twonn"]
    print(f"\nPR - TwoNN gap:  real {gap_real:.1f}   matched gaussian {gap_gauss:.1f}")
    if P["gaussian_twonn"] > 2.0 * P["real_twonn"]:
        print("VERDICT: the low TwoNN is not an estimator artifact; the matched Gaussian with the "
              "same linear spectrum reads substantially higher. Curvature is real.")
    else:
        print("VERDICT: WARNING. The matched Gaussian reads nearly as low as the real data. The "
              "5-6 figure is dominated by estimator bias at this sample size, and the manifold "
              "claim in Section 4.1 needs to be softened or dropped.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
