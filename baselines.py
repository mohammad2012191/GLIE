"""Training-free baselines, all scored by the shared harness in metrics.py.

  ceiling         uncompressed 1024 patches. Nothing can beat it; every table is read against it.
  raw_kmeans      per-page k-means centroids as-is. The naive baseline.
  norm_kmeans     the same centroids L2-normalized back onto the sphere. Free, and much stronger.
  cluster_merge   semantic agglomerative merge (cosine, average linkage) + renormalization. This is
                  Light-ColPali's merging strategy with its fine-tuning removed, so it isolates
                  what their MERGE contributes from what their FINE-TUNE contributes -- a
                  decomposition their paper reports only in combination.
  token_pool      sequential token pooling, the training-free incumbent.

The main table's baseline column should be max(norm_kmeans, cluster_merge) per budget, so the
comparison can never be called cherry-picked.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


def raw_kmeans(artifact):
    return artifact["centers"].float()


def norm_kmeans(artifact):
    return F.normalize(artifact["centers"].float(), dim=-1)


def cluster_merge(corpus, k, cache_path=None):
    """Agglomerative cosine merge to k vectors per page, renormalized to the sphere."""
    import os
    if cache_path and os.path.exists(cache_path):
        c = torch.load(cache_path)
        if c.shape[0] == corpus.n_pages and c.shape[1] == k:
            return c

    out = []
    for i in range(corpus.n_pages):
        x = corpus.page[i][corpus.pmask[i]]
        xn = F.normalize(x, dim=-1).numpy().astype(np.float64)
        n = xn.shape[0]
        if k >= n:
            merged = torch.from_numpy(xn).float()
        else:
            d = 1.0 - xn @ xn.T
            d = (d + d.T) / 2.0
            np.fill_diagonal(d, 0.0)
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                          linkage="average").fit_predict(d)
            lab = torch.from_numpy(lab).long()
            merged = torch.stack([x[lab == j].mean(0) if (lab == j).any() else x[0]
                                  for j in range(k)])
        merged = F.normalize(merged, dim=-1)
        if merged.shape[0] < k:
            merged = torch.cat([merged, merged[-1:].expand(k - merged.shape[0], -1)], 0)
        out.append(merged[:k])
        if (i + 1) % 100 == 0:
            print(f"    cluster_merge {corpus.slug} k={k}: {i + 1}/{corpus.n_pages}")

    reps = torch.stack(out)
    if cache_path:
        torch.save(reps, cache_path)
    return reps


def token_pool(corpus, k):
    """Sequential mean pooling of adjacent patches into k groups, renormalized.

    The cheapest training-free reduction and the one every practitioner reaches for first.
    """
    out = []
    for i in range(corpus.n_pages):
        x = corpus.page[i][corpus.pmask[i]]
        idx = torch.linspace(0, x.shape[0], k + 1).long()
        groups = [x[idx[j]:max(idx[j + 1], idx[j] + 1)].mean(0) for j in range(k)]
        out.append(F.normalize(torch.stack(groups), dim=-1))
    return torch.stack(out)


def all_static_baselines(corpus, artifact, k, cache_dir=None):
    """name -> (reps, rmask). rmask is None for fixed-width codes."""
    import os
    cm_cache = os.path.join(cache_dir, f"{corpus.slug}__merge_k{k}.pt") if cache_dir else None
    return {
        "raw_kmeans": (raw_kmeans(artifact), None),
        "norm_kmeans": (norm_kmeans(artifact), None),
        "cluster_merge": (cluster_merge(corpus, k, cm_cache), None),
        "token_pool": (token_pool(corpus, k), None),
    }
