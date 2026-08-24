"""THE eval harness. Every method in the paper is scored by these functions and no others.

If two methods are ever scored by different code, the main table is worthless, so scoring lives
in exactly one place. Two entry points:

    evaluate_static(...)   any method that produces a fixed set of vectors per page
                           (raw k-means, normalized k-means, cluster-merge, Light-ColPali,
                           GLIE stage 1, the uncompressed ceiling)
    evaluate_cascade(...)  a two-stage method: shortlist on the code, rescore with a decoder

Metric family: nDCG@{1,5,10}, MRR, Recall@{1,5,10}. Single relevant page per query, which is the
ViDoRe v1 structure. PRIMARY_METRIC in config.py picks the headline column.
"""
import math

import torch

CAND_OFFSET = 1e3   # keeps a rescored shortlist above the un-rescored tail


def maxsim(q, qm, p, pm=None):
    """Late-interaction score. q:[B,Q,D] qm:[B,Q] p:[C,P,D] pm:[C,P] -> [B,C].

    Padded PATCHES are masked to -inf before the max (otherwise a zero row can win the max and
    invent evidence). Padded QUERY tokens contribute 0 to the sum. The division by query length
    is a per-query constant, so it cannot change any ranking; it is kept because the training
    losses compare these values across queries.
    """
    sim = torch.einsum("bqd,cpd->bcqp", q, p)
    if pm is not None:
        sim = sim.masked_fill(~pm[None, :, None, :], float("-inf"))
    sim = sim.amax(3).masked_fill(~qm[:, None, :], 0.0)
    return sim.sum(2) / qm.sum(1).clamp(min=1.0)[:, None]


def rank_of_gold(scores, gold):
    """scores:[Q,N] gold:[Q] -> 1-indexed rank of the gold doc for each query."""
    order = scores.argsort(dim=1, descending=True)
    hits = (order == gold[:, None]).nonzero(as_tuple=False)
    # exactly one hit per row, and nonzero() returns row-major, so column 1 aligns with rows
    return hits[:, 1].float() + 1.0


def metrics_from_ranks(ranks):
    def ndcg_at(k):
        return float(torch.where(ranks <= k, 1.0 / torch.log2(ranks + 1), torch.zeros_like(ranks)).mean())

    def recall_at(k):
        return float((ranks <= k).float().mean())

    return {
        "ndcg@1": ndcg_at(1), "ndcg@5": ndcg_at(5), "ndcg@10": ndcg_at(10),
        "mrr": float((1.0 / ranks).mean()),
        "recall@1": recall_at(1), "recall@5": recall_at(5), "recall@10": recall_at(10),
        "mean_rank": float(ranks.mean()), "median_rank": float(ranks.median()),
        "n_queries": int(ranks.numel()),
    }


@torch.no_grad()
def score_matrix(reps, rmask, qemb, qmask, q_idx, device, q_chunk=32, d_chunk=128):
    """[len(q_idx), N] score matrix, streamed so a 1024-vector corpus still fits."""
    reps = reps.float()
    out = torch.empty(len(q_idx), reps.shape[0])
    for s in range(0, len(q_idx), q_chunk):
        qi = q_idx[s:s + q_chunk]
        q = qemb[qi].to(device); qm = qmask[qi].to(device)
        for t in range(0, reps.shape[0], d_chunk):
            rm = rmask[t:t + d_chunk].to(device) if rmask is not None else None
            out[s:s + len(qi), t:t + d_chunk] = maxsim(q, qm, reps[t:t + d_chunk].to(device), rm).cpu()
    return out


@torch.no_grad()
def evaluate_static(reps, rmask, corpus, q_idx, device):
    """Score a fixed per-page representation. `reps` is [N, *, D] over the WHOLE corpus."""
    scores = score_matrix(reps, rmask, corpus.qemb, corpus.qmask, q_idx, device)
    gold = corpus.gold[q_idx]
    return metrics_from_ranks(rank_of_gold(scores, gold)), scores


@torch.no_grad()
def evaluate_cascade(code_all, decode_fn, corpus, q_idx, device, topl=20,
                     tail_policy="stage1", oracle=False, q_chunk=16):
    """Two-stage: shortlist on `code_all`, rescore the shortlist with `decode_fn`.

    decode_fn(doc_indices) -> [len(doc_indices), P, D] regenerated vectors on `device`.
    oracle=True rescores with the TRUE page vectors instead.

    NOTE ON THE ORACLE. It is a REFERENCE POINT, not an upper bound. It reports what you get by
    rescoring the shortlist with uncompressed vectors, using the same ranking function as the
    ceiling -- but a decoder whose distortions happen to reorder the shortlist favourably can beat
    it, and on small eval sets several cells do. Likewise the decoder can exceed the FULL-CORPUS
    ceiling, because a shortlist that retains the gold page makes it compete against topl-1
    distractors instead of N-1. Neither is a leak; both are properties of a cascade, and on
    20-query subsets both are usually one query moving one rank.

    tail_policy:
      "stage1" -- non-candidates keep their stage-1 score, candidates are fused above them.
                  A shortlist miss still gets whatever rank stage 1 gave it (partial credit).
      "strict" -- non-candidates are pushed below everything, so a shortlist miss scores 0.
    """
    gold = corpus.gold[q_idx]
    N = code_all.shape[0]
    all_ranks, hits = [], 0

    for s in range(0, len(q_idx), q_chunk):
        qi = q_idx[s:s + q_chunk]
        q = corpus.qemb[qi].to(device); qm = corpus.qmask[qi].to(device)
        g = gold[s:s + q_chunk]
        A = len(qi)

        stage1 = torch.empty(A, N)
        for t in range(0, N, 128):
            stage1[:, t:t + 128] = maxsim(q, qm, code_all[t:t + 128].to(device)).cpu()
        cand = stage1.topk(min(topl, N), dim=1).indices                      # [A, topl]

        final = stage1.clone() if tail_policy == "stage1" else torch.full_like(stage1, -1e9)
        for a in range(A):
            c = cand[a]
            if oracle:
                dec = corpus.page[c].to(device)
                dm = corpus.pmask[c].to(device)
            else:
                dec = decode_fn(c)
                dm = None
            s2 = maxsim(q[a:a + 1], qm[a:a + 1], dec, dm)[0].cpu()
            final[a, c] = s2 + CAND_OFFSET

        all_ranks.append(rank_of_gold(final, g))
        hits += int((cand == g[:, None]).any(dim=1).sum())

    m = metrics_from_ranks(torch.cat(all_ranks))
    m[f"shortlist_recall@{topl}"] = hits / len(q_idx)
    return m


# ---------------------------------------------------------------------------
# Storage accounting. Reported in the paper, so it is computed, never asserted.
# ---------------------------------------------------------------------------
def storage_bytes(k, dim, method):
    """Bytes stored PER PAGE at budget k.

    vectors      k*dim at bf16/fp16 (2 bytes)
    norms        GLIE keeps one scalar per cluster (the centroid magnitude) at fp16
    counts       GLIE keeps one cluster size per cluster; <=1024 fits in uint16
    Everything else (the decoder weights) is a single shared module, amortized over the whole
    corpus, and is reported separately as `shared_module_bytes` -- never per page.
    """
    vec = k * dim * 2
    extra = 0
    if method.startswith("glie"):
        extra = k * 2 + k * 2          # fp16 norm + uint16 count per cluster
    return {
        "vectors_per_page": k,
        "bytes_per_page": vec + extra,
        "bytes_vectors": vec,
        "bytes_sidecar": extra,
        "sidecar_overhead_pct": round(100.0 * extra / max(vec, 1), 2),
    }


def compression_vs_full(k, n_patches):
    return round(n_patches / max(k, 1), 2)
