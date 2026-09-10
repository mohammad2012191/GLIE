"""ViDoRe v2 evaluation: multi-relevant qrels, zero-shot, same frozen codec as the main table.

ViDoRe v2 ships BEIR-style: separate `corpus` (images), `queries` (text), and `qrels`
(query-id, corpus-id, score) configs, with MULTIPLE relevant pages per query. The v1 harness
scores single-gold ranks, so this script carries its own graded-nDCG layer and reuses
everything else: the encoder loader, per-page k-means artifacts, the GLIE model, MaxSim,
and the decode function. The codec is NOT retrained: it loads the std_main checkpoints
fitted on colpali_train_set, so v2 is a pure zero-shot generalization test.

Run from the parent of glie/:

  # 1. FIRST: verify the dataset schema (column names shift between releases)
  python -m glie.vidore_v2 --inspect

  # 2. full eval
  python -m glie.vidore_v2 \
      --cache_dir ./cache_v2 \
      --ckpt_dir  ./out/checkpoints \
      --out_dir   ./out/results \
      --codes 2 4 8 16 32 64 --tag v2_std

Metric note: graded nDCG@5 with linear gain, matching pytrec_eval's default. The official
vidore-benchmark evaluator remains the camera-ready arbiter; this script is for the
research table and uses one scoring path for every method.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .baselines import all_static_baselines
from .data import Corpus, _encoder_classes, kmeans_artifact
from .metrics import maxsim, score_matrix, storage_bytes, compression_vs_full, CAND_OFFSET
from .models import glie_from_state_dict
from .train_glie import encode_corpus, make_decode_fn

VIDORE_V2 = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_v2",
    "vidore/economics_reports_v2",
    "vidore/esg_reports_human_labeled_v2",
]

# Column-name candidates across dataset releases.
CORPUS_ID = ["corpus-id", "corpus_id", "docid", "doc-id", "id"]
QUERY_ID = ["query-id", "query_id", "qid", "id"]
QUERY_TEXT = ["query", "text", "question"]
SCORE = ["score", "relevance", "is-relevant", "answer"]


def _col(ds, cands, what):
    for c in cands:
        if c in ds.column_names:
            return c
    raise KeyError(f"no {what} column among {cands}; found {ds.column_names} -- "
                   f"run with --inspect and adjust the candidate lists")


def _only_split(dd):
    if hasattr(dd, "column_names") and not isinstance(dd.column_names, dict):
        return dd
    keys = list(dd.keys())
    if "test" in keys:
        return dd["test"]
    return dd[keys[0]]


def load_v2(name):
    """-> images(list), image_ids(list), queries(list), query_ids(list), qrels{qid: {cid: rel}}"""
    from datasets import load_dataset
    corpus = _only_split(load_dataset(name, "corpus"))
    queries = _only_split(load_dataset(name, "queries"))
    qrels = _only_split(load_dataset(name, "qrels"))

    cid = _col(corpus, CORPUS_ID, "corpus-id")
    qid = _col(queries, QUERY_ID, "query-id")
    qtx = _col(queries, QUERY_TEXT, "query text")
    r_q = _col(qrels, QUERY_ID, "qrels query-id")
    r_c = _col(qrels, CORPUS_ID, "qrels corpus-id")
    r_s = _col(qrels, SCORE, "qrels score")

    rel = {}
    for row in qrels:
        s = float(row[r_s])
        if s > 0:
            rel.setdefault(row[r_q], {})[row[r_c]] = s

    q_ids, q_texts = [], []
    for row in queries:
        if row[qid] in rel and row[qtx] and str(row[qtx]).strip():
            q_ids.append(row[qid]); q_texts.append(str(row[qtx]))

    images = [row["image"].convert("RGB") for row in corpus]
    image_ids = [row[cid] for row in corpus]
    return images, image_ids, q_texts, q_ids, rel


def encode_v2(name, slug, cache_dir, model_name, encode_batch, device):
    """Encode corpus + queries once, cache as {slug}__*.pt. Returns (Corpus, qrels_matrix)."""
    paths = {s: os.path.join(cache_dir, f"{slug}__{s}.pt")
             for s in ["page", "pmask", "qemb", "qmask", "rels", "meta"]}
    if all(os.path.exists(p) for p in paths.values()):
        meta = torch.load(paths["meta"])
        if meta.get("model") == model_name:
            corpus = Corpus(slug=slug, page=torch.load(paths["page"]),
                            pmask=torch.load(paths["pmask"]), qemb=torch.load(paths["qemb"]),
                            qmask=torch.load(paths["qmask"]),
                            gold=torch.zeros(torch.load(paths["qemb"]).shape[0], dtype=torch.long),
                            meta=meta)
            return corpus, torch.load(paths["rels"])

    images, image_ids, q_texts, q_ids, rel = load_v2(name)
    n, nq = len(images), len(q_texts)
    print(f"  [extract] {slug}: {n} corpus pages, {nq} queries with positive qrels")

    Model, Processor = _encoder_classes(model_name)
    model = Model.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                  device_map=device).eval()
    processor = Processor.from_pretrained(model_name)

    pe = []
    with torch.no_grad():
        for s in range(0, n, encode_batch):
            b = processor.process_images(images[s:s + encode_batch]).to(device)
            out = model(**b)
            pe.extend(out[i].float().cpu() for i in range(out.shape[0]))
    P = max(e.shape[0] for e in pe)
    page = torch.zeros(n, P, pe[0].shape[-1]); pmask = torch.zeros(n, P, dtype=torch.bool)
    for i, e in enumerate(pe):
        page[i, :e.shape[0]] = e; pmask[i, :e.shape[0]] = True

    qe, qm = [], []
    with torch.no_grad():
        for s in range(0, nq, encode_batch):
            b = processor.process_queries(q_texts[s:s + encode_batch]).to(device)
            out = model(**b)
            am = b["attention_mask"].bool().cpu()
            qe.extend(out[i].float().cpu() for i in range(out.shape[0]))
            qm.extend(am[i] for i in range(am.shape[0]))
    L = max(m.shape[0] for m in qm)
    qemb = torch.zeros(nq, L, page.shape[-1]); qmask = torch.zeros(nq, L, dtype=torch.bool)
    for i, (e, m) in enumerate(zip(qe, qm)):
        qemb[i, :e.shape[0]] = e; qmask[i, :m.shape[0]] = m
    qmask &= qemb.norm(dim=-1) > 1e-8

    # graded relevance matrix [nq, n]
    id_to_col = {c: j for j, c in enumerate(image_ids)}
    rels = torch.zeros(nq, n)
    dropped = 0
    for i, q in enumerate(q_ids):
        for c, s in rel[q].items():
            if c in id_to_col:
                rels[i, id_to_col[c]] = s
            else:
                dropped += 1
    if dropped:
        print(f"  [warn] {dropped} qrel entries reference corpus-ids absent from the corpus config")

    meta = {"dataset": name, "model": model_name, "n": n, "nq": nq,
            "P": int(page.shape[1]), "D": int(page.shape[2])}
    torch.save(page, paths["page"]); torch.save(pmask, paths["pmask"])
    torch.save(qemb, paths["qemb"]); torch.save(qmask, paths["qmask"])
    torch.save(rels, paths["rels"]); torch.save(meta, paths["meta"])
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    corpus = Corpus(slug=slug, page=page, pmask=pmask, qemb=qemb, qmask=qmask,
                    gold=torch.zeros(nq, dtype=torch.long), meta=meta)
    return corpus, rels


# ---------------------------------------------------------------------------
# Multi-relevant metrics: ONE scoring path for every method, like the v1 harness.
# ---------------------------------------------------------------------------
def graded_metrics(scores, rels):
    """scores:[Q,N] rels:[Q,N] graded (0 = not relevant). Linear-gain nDCG."""
    out = {}
    order = scores.argsort(dim=1, descending=True)
    gains = torch.gather(rels, 1, order)                       # relevance in ranked order
    ideal = rels.sort(dim=1, descending=True).values
    disc = 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2.0)
    for k in [1, 5, 10]:
        dcg = (gains[:, :k] * disc[:k]).sum(1)
        idcg = (ideal[:, :k] * disc[:k]).sum(1).clamp(min=1e-9)
        out[f"ndcg@{k}"] = float((dcg / idcg).mean())
        found = (gains[:, :k] > 0).float().sum(1)
        total = (rels > 0).float().sum(1).clamp(min=1.0)
        out[f"recall@{k}"] = float((found / torch.minimum(total, torch.tensor(float(k)))).mean())
    first = (gains > 0).float().argmax(dim=1).float() + 1.0
    none = (gains > 0).sum(1) == 0
    rr = torch.where(none, torch.zeros_like(first), 1.0 / first)
    out["mrr"] = float(rr.mean())
    out["n_queries"] = int(scores.shape[0])
    return out


@torch.no_grad()
def cascade_scores(code_all, decode_fn, corpus, device, topl, oracle=False, q_chunk=16):
    """Full [Q,N] fused score matrix: stage-1 everywhere, rescored shortlist offset above."""
    N, Q = code_all.shape[0], corpus.n_queries
    final = torch.empty(Q, N)
    for s in range(0, Q, q_chunk):
        qi = np.arange(s, min(Q, s + q_chunk))
        q = corpus.qemb[qi].to(device); qm = corpus.qmask[qi].to(device)
        stage1 = torch.empty(len(qi), N)
        for t in range(0, N, 128):
            stage1[:, t:t + 128] = maxsim(q, qm, code_all[t:t + 128].to(device)).cpu()
        cand = stage1.topk(min(topl, N), dim=1).indices
        fused = stage1.clone()
        for a in range(len(qi)):
            c = cand[a]
            if oracle:
                dec, dm = corpus.page[c].to(device), corpus.pmask[c].to(device)
            else:
                dec, dm = decode_fn(c), None
            fused[a, c] = maxsim(q[a:a + 1], qm[a:a + 1], dec, dm)[0].cpu() + CAND_OFFSET
        final[s:s + len(qi)] = fused
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=VIDORE_V2)
    ap.add_argument("--codes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--model_name", default="vidore/colpali-v1.3")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True,
                    help="folder holding colpali_train_set__train_k{k}_{ckpt_tag}.pt")
    ap.add_argument("--ckpt_tag", default="std_main")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="v2_std")
    ap.add_argument("--encode_batch", type=int, default=4)
    ap.add_argument("--topl", type=int, default=20)
    ap.add_argument("--inspect", action="store_true",
                    help="print each dataset's configs/columns and exit, encoding nothing")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.inspect:
        from datasets import get_dataset_config_names, load_dataset
        for name in args.datasets:
            print(f"\n=== {name} ===")
            try:
                cfgs = get_dataset_config_names(name)
                print("configs:", cfgs)
                for c in cfgs:
                    ds = _only_split(load_dataset(name, c))
                    print(f"  [{c}] rows={len(ds)} columns={ds.column_names}")
            except Exception as e:
                print("  FAILED:", e)
        return

    os.makedirs(args.cache_dir, exist_ok=True); os.makedirs(args.out_dir, exist_ok=True)
    rows, blob = [], {}
    for name in args.datasets:
        slug = name.split("/")[-1]
        corpus, rels = encode_v2(name, slug, args.cache_dir, args.model_name,
                                 args.encode_batch, device)
        print(f"\n=== {slug}: {corpus.n_pages} pages, {corpus.n_queries} queries "
              f"(multi-relevant, mean {float((rels > 0).sum(1).float().mean()):.1f} "
              f"rel/query), P={corpus.n_patches} D={corpus.dim}")

        ceil_scores = score_matrix(corpus.page, corpus.pmask, corpus.qemb, corpus.qmask,
                                   np.arange(corpus.n_queries), device)
        ceil = graded_metrics(ceil_scores, rels)
        print(f"    ceiling  ndcg@5={ceil['ndcg@5']:.4f}")
        rows.append({"dataset": slug, "k": corpus.n_patches, "method": "ceiling", **ceil,
                     **storage_bytes(corpus.n_patches, corpus.dim, "full"), "compression": 1.0})

        for k in args.codes:
            print(f"\n  --- {slug} k={k} ---")
            art = kmeans_artifact(corpus, k, args.cache_dir)

            for mname, (reps, rmask) in all_static_baselines(corpus, art, k,
                                                             args.cache_dir).items():
                sc = score_matrix(reps, rmask, corpus.qemb, corpus.qmask,
                                  np.arange(corpus.n_queries), device)
                m = graded_metrics(sc, rels)
                rows.append({"dataset": slug, "k": k, "method": mname, **m,
                             **storage_bytes(k, corpus.dim, mname),
                             "compression": compression_vs_full(k, corpus.n_patches)})
                print(f"    {mname:<16}ndcg@5={m['ndcg@5']:.4f}")

            ckpt = os.path.join(args.ckpt_dir,
                                f"colpali_train_set__train_k{k}_{args.ckpt_tag}.pt")
            saved = torch.load(ckpt, map_location="cpu")
            model = glie_from_state_dict(corpus.dim, saved["state_dict"]).to(device)

            model.load_state_dict(saved["stage1_state_dict"]); model.eval()
            s1_reps, _ = encode_corpus(model, corpus, art, device)
            m1 = graded_metrics(score_matrix(s1_reps, None, corpus.qemb, corpus.qmask,
                                             np.arange(corpus.n_queries), device), rels)

            model.load_state_dict(saved["state_dict"]); model.eval()
            ref_all, nrm_all = encode_corpus(model, corpus, art, device)
            dec_fn = make_decode_fn(model, corpus, art, ref_all, nrm_all, device)
            md = graded_metrics(cascade_scores(ref_all, dec_fn, corpus, device, args.topl),
                                rels)
            mo = graded_metrics(cascade_scores(ref_all, dec_fn, corpus, device, args.topl,
                                               oracle=True), rels)

            for mname, m in [("glie_stage1", m1), ("glie_decoder", md), ("glie_oracle", mo)]:
                rows.append({"dataset": slug, "k": k, "method": mname, **m,
                             **storage_bytes(k, corpus.dim, "glie"),
                             "compression": compression_vs_full(k, corpus.n_patches)})
            print(f"    {'glie_stage1':<16}ndcg@5={m1['ndcg@5']:.4f}")
            print(f"    {'glie_decoder':<16}ndcg@5={md['ndcg@5']:.4f}  "
                  f"(oracle {mo['ndcg@5']:.4f})")

            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.out_dir, f"{args.tag}__main_table.csv"), index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"VIDORE V2 (graded ndcg@5), averaged over {len(args.datasets)} dataset(s)")
    print("=" * 96)
    piv = (df[df.method != "ceiling"]
           .pivot_table(index="method", columns="k", values="ndcg@5", aggfunc="mean"))
    order = ["raw_kmeans", "token_pool", "cluster_merge", "norm_kmeans",
             "glie_stage1", "glie_decoder", "glie_oracle"]
    print(piv.reindex([m for m in order if m in piv.index]).round(4).to_string())
    print(f"\nceiling = {df[df.method == 'ceiling']['ndcg@5'].mean():.4f}")

    TF = ["norm_kmeans", "cluster_merge", "token_pool"]
    tf = df[df.method.isin(TF)]
    best = tf.groupby(["dataset", "k"])["ndcg@5"].max()
    print("\nmargin of glie_decoder over the BEST training-free baseline:")
    g = df[df.method == "glie_decoder"].set_index(["dataset", "k"])["ndcg@5"]
    common = g.index.intersection(best.index)
    margin = (g.loc[common] - best.loc[common]).groupby("k").mean()
    for k in sorted(margin.index):
        print(f"    k={k:<4} {margin[k]:+.4f}")
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
