"""Corpus extraction, caching, the split, and per-page k-means artifacts.

Two invariants that protect every downstream number:

1. THE CORPUS IS EVERY PAGE. Rows whose query is missing stay in the index as distractors and are
   simply excluded from the query set. Dropping those pages (an easy mistake) silently shrinks the
   corpus on the shift/synthetic subsets and inflates every method's score.

2. THE SPLIT IS A FUNCTION OF (n, seed) ONLY. Every method calls the same `make_split`, so no
   method can be evaluated on a different holdout than another.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class Corpus:
    slug: str
    page: torch.Tensor      # [N, P, D] float32, L2-normalized by the encoder
    pmask: torch.Tensor     # [N, P] bool
    qemb: torch.Tensor      # [Q, L, D] float32
    qmask: torch.Tensor     # [Q, L] bool
    gold: torch.Tensor      # [Q] long, index of the relevant page for each query
    meta: dict

    @property
    def n_pages(self):
        return self.page.shape[0]

    @property
    def n_queries(self):
        return self.qemb.shape[0]

    @property
    def dim(self):
        return self.page.shape[2]

    @property
    def n_patches(self):
        return self.page.shape[1]


def make_split(n_queries, seed=0, holdout_frac=0.2):
    """Identical for every method. Returns (train_q, holdout_q) as index arrays into the QUERY axis."""
    perm = np.random.default_rng(seed).permutation(n_queries)
    n_hold = max(1, int(round(holdout_frac * n_queries)))
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _paths(cache_dir, slug):
    j = lambda s: os.path.join(cache_dir, f"{slug}__{s}.pt")
    return dict(page=j("page"), pmask=j("pmask"), qemb=j("qemb"), qmask=j("qmask"),
                gold=j("gold"), meta=os.path.join(cache_dir, f"{slug}__meta.json"))


def _image_key(img):
    """Identity of a page image, for de-duplication. Exact-match on pixel content."""
    return (img.size, hash(img.convert("RGB").tobytes()))


def _encoder_classes(model_name):
    """Pick the colpali_engine model/processor pair from the checkpoint name. Everything downstream
    is encoder-agnostic: pages of any length are padded under pmask, and D is read from the output."""
    if "colqwen2.5" in model_name:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        return ColQwen2_5, ColQwen2_5_Processor
    if "colqwen" in model_name:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        return ColQwen2, ColQwen2Processor
    from colpali_engine.models import ColPali, ColPaliProcessor
    return ColPali, ColPaliProcessor


def _extract(dataset, slug, cache_dir, model_name, encode_batch, max_pages, device, split="test"):
    from datasets import load_dataset

    Model, Processor = _encoder_classes(model_name)
    print(f"  [extract] {dataset} [{split}] with {model_name}")
    model = Model.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
    processor = Processor.from_pretrained(model_name)
    # `split[:N]` keeps the download bounded -- essential for colpali_train_set, which is ~118k rows
    spec = f"{split}[:{max_pages}]" if max_pages else split
    ds = load_dataset(dataset, split=spec)

    rows = list(range(len(ds)))
    if max_pages:
        rows = rows[:max_pages]

    # ---- DE-DUPLICATE THE CORPUS BY IMAGE IDENTITY -------------------------------------------
    # Several ViDoRe subsets repeat the SAME page across many queries: TAT-DQA is 1663 rows over
    # ~277 unique images, TabFQuAD 280 rows over ~70. Indexing every row as a separate page puts
    # identical copies in the corpus, and then the "correct" page is unrecoverable: the copies
    # score identically and argsort breaks the tie arbitrarily, so even the UNCOMPRESSED ceiling
    # collapses (0.29 on TAT-DQA). The official evaluator aggregates per image_filename; we index
    # each distinct image once and point every one of its queries at that single entry.
    # Telltale sign that this was happening: the oracle scoring BELOW the decoder, because true
    # page vectors are byte-identical across duplicates and cannot separate them at all.
    has_fname = "image_filename" in getattr(ds, "column_names", [])
    seen, uniq_rows, row_to_page = {}, [], {}
    for r in rows:
        key = ds[r]["image_filename"] if has_fname else _image_key(ds[r]["image"])
        if key not in seen:
            seen[key] = len(uniq_rows)
            uniq_rows.append(r)
        row_to_page[r] = seen[key]

    images = [ds[r]["image"].convert("RGB") for r in uniq_rows]
    q_rows = [r for r in rows if ds[r]["query"] is not None and str(ds[r]["query"]).strip()]
    queries = [ds[r]["query"] for r in q_rows]
    gold = torch.tensor([row_to_page[r] for r in q_rows], dtype=torch.long)
    n, nq = len(images), len(queries)
    dup = len(rows) - n
    print(f"  [extract] corpus {n} unique pages | {nq} queries "
          f"({dup} duplicate rows collapsed"
          f"{', by image_filename' if has_fname else ', by pixel content'})")
    if dup:
        print(f"  [extract] NOTE: {len(rows)} rows -> {n} pages; without this de-duplication the "
              f"uncompressed ceiling on this subset would be badly understated.")

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
            b = processor.process_queries(queries[s:s + encode_batch]).to(device)
            out = model(**b)
            am = b["attention_mask"].bool().cpu()
            qe.extend(out[i].float().cpu() for i in range(out.shape[0]))
            qm.extend(am[i] for i in range(am.shape[0]))
    L = max(m.shape[0] for m in qm)
    qemb = torch.zeros(nq, L, page.shape[-1]); qmask = torch.zeros(nq, L, dtype=torch.bool)
    for i, (e, m) in enumerate(zip(qe, qm)):
        qemb[i, :e.shape[0]] = e; qmask[i, :m.shape[0]] = m
    qmask &= qemb.norm(dim=-1) > 1e-8            # drop all-zero query tokens

    p = _paths(cache_dir, slug)
    torch.save(page, p["page"]); torch.save(pmask, p["pmask"])
    torch.save(qemb, p["qemb"]); torch.save(qmask, p["qmask"]); torch.save(gold, p["gold"])
    json.dump({"dataset": dataset, "model": model_name, "n": n, "nq": nq, "dedup": True,
               "max_pages": int(max_pages), "rows": len(rows), "split": split,
               "P": int(page.shape[1]), "D": int(page.shape[2])}, open(p["meta"], "w"))
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()


def load_corpus(dataset, cache_dir, model_name, encode_batch=8, max_pages=0, device="cuda",
                split="test"):
    # a non-test split is a different corpus, so it gets its own cache slot
    slug = dataset.split("/")[-1] + ("" if split == "test" else f"__{split}")
    p = _paths(cache_dir, slug)
    need = not all(os.path.exists(v) for v in p.values())
    if not need:
        m = json.load(open(p["meta"]))
        # `dedup`, `max_pages` and `split` are part of the fingerprint: a cache written before the
        # de-duplication fix, or under a different page cap or split, describes a different corpus
        # and must never be silently reused.
        if (m.get("dataset") != dataset or m.get("model") != model_name
                or not m.get("dedup", False) or int(m.get("max_pages", 0)) != int(max_pages)
                or m.get("split", "test") != split):
            print(f"  [cache] fingerprint mismatch for {slug} (dedup={m.get('dedup', False)}, "
                  f"max_pages={m.get('max_pages')}, split={m.get('split', 'test')}), re-extracting")
            need = True
    if need:
        os.makedirs(cache_dir, exist_ok=True)
        _extract(dataset, slug, cache_dir, model_name, encode_batch, max_pages, device, split)

    return Corpus(
        slug=slug,
        page=torch.load(p["page"]).float(),
        pmask=torch.load(p["pmask"]).bool(),
        qemb=torch.load(p["qemb"]).float(),
        qmask=torch.load(p["qmask"]).bool(),
        gold=torch.load(p["gold"]).long(),
        meta=json.load(open(p["meta"])),
    )


# ---------------------------------------------------------------------------
# Per-page k-means + the slot layout the decoder needs
# ---------------------------------------------------------------------------
def build_slot_layout(counts, n_slots):
    """counts:[N,k] -> per-page assignment of exactly `n_slots` output slots to clusters.

    Cluster j owns counts[j] slots, so the decoder reproduces the real cluster structure. Slot 0
    of each cluster is its ANCHOR: the decoder emits the refined vector there verbatim, which makes
    the decoded set a superset of the code set.

    Empty clusters are the edge case that breaks a naive implementation (ragged rows, and a cluster
    with no anchor). Every cluster is therefore guaranteed at least one slot, and the surplus or
    deficit is absorbed by the largest cluster so each page has exactly n_slots.
    """
    N, k = counts.shape
    owner = torch.zeros(N, n_slots, dtype=torch.long)
    rank = torch.zeros(N, n_slots, dtype=torch.long)
    frac = torch.zeros(N, n_slots, dtype=torch.float32)
    anchor = torch.zeros(N, n_slots, dtype=torch.bool)

    for i in range(N):
        c = counts[i].clone().long()
        c = c.clamp(min=1)                                   # every cluster gets an anchor
        diff = int(n_slots - c.sum())
        if diff != 0:                                        # absorb into the largest cluster
            j = int(c.argmax())
            c[j] = max(1, c[j] + diff)
            diff = int(n_slots - c.sum())
            if diff != 0:                                    # pathological k > n_slots
                c[j] = max(1, c[j] + diff)
        o, r, f, a = [], [], [], []
        for j, cnt in enumerate(c.tolist()):
            cnt = max(1, int(cnt))
            o.extend([j] * cnt); r.extend(range(cnt))
            denom = max(1, cnt - 1)
            f.extend([t / denom for t in range(cnt)])
            a.extend([True] + [False] * (cnt - 1))
        o, r, f, a = o[:n_slots], r[:n_slots], f[:n_slots], a[:n_slots]
        owner[i] = torch.tensor(o); rank[i] = torch.tensor(r)
        frac[i] = torch.tensor(f); anchor[i] = torch.tensor(a)
    return {"slot_owner": owner, "slot_rank": rank, "slot_fraction": frac, "anchor_mask": anchor}


def kmeans_artifact(corpus, k, cache_dir, n_init=2, seed=0):
    """Per-page k-means. Cached, because it is by far the slowest non-GPU step."""
    f = os.path.join(cache_dir, f"{corpus.slug}__kmeans_k{k}.pt")
    if os.path.exists(f):
        art = torch.load(f)
        if art["centers"].shape[0] == corpus.n_pages:
            return art

    centers, counts, labels, target_rank = [], [], [], []
    for i in range(corpus.n_pages):
        pts = corpus.page[i][corpus.pmask[i]].numpy()
        est = KMeans(n_clusters=min(k, len(pts)), n_init=n_init, random_state=seed).fit(pts)
        lab = torch.from_numpy(est.labels_).long()
        cen = torch.from_numpy(est.cluster_centers_).float()
        if cen.shape[0] < k:                                 # fewer points than clusters
            cen = torch.cat([cen, cen[-1:].expand(k - cen.shape[0], -1)], 0)
        centers.append(cen)
        counts.append(torch.bincount(lab, minlength=k)[:k])
        # rank of each patch within its own cluster, used to align the set loss
        rk = torch.empty_like(lab)
        for j in range(k):
            pos = (lab == j).nonzero(as_tuple=False).flatten()
            rk[pos] = torch.arange(len(pos))
        labels.append(lab); target_rank.append(rk)
        if (i + 1) % 100 == 0:
            print(f"    kmeans {corpus.slug} k={k}: {i + 1}/{corpus.n_pages}")

    P = corpus.n_patches
    lab_pad = torch.zeros(corpus.n_pages, P, dtype=torch.long)
    rk_pad = torch.zeros(corpus.n_pages, P, dtype=torch.long)
    for i, (l, r) in enumerate(zip(labels, target_rank)):
        lab_pad[i, :len(l)] = l; rk_pad[i, :len(r)] = r

    cnt = torch.stack(counts)
    art = {"centers": torch.stack(centers), "counts": cnt,
           "labels": lab_pad, "target_rank": rk_pad,
           **build_slot_layout(cnt, P)}
    torch.save(art, f)
    return art
