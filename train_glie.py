"""GLIE training: refine the direction, keep the magnitude, regenerate on demand.

Leak discipline (every one of these is load-bearing):
  * hard negatives are mined from TRAIN queries against TRAIN documents only
  * the decoder is QUERY-FREE, so nothing about an eval query can enter a representation
  * the teacher (fine-vector) scores touch the model only through train-side mining and the loss
    targets of train queries
  * model selection uses the holdout split, and the reported number comes from the best checkpoint
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import losses as L
from .data import make_split
from .metrics import evaluate_cascade, evaluate_static, storage_bytes, compression_vs_full
from .models import GLIE


def subsample_train(train_q, cfg):
    """Keep only --train_frac of the training queries. Holdout and corpus are untouched, so points
    on a data-scaling curve stay directly comparable to each other and to the full-data run."""
    frac = getattr(cfg, "train_frac", 1.0)
    if frac >= 1.0:
        return train_q
    n = max(1, int(round(frac * len(train_q))))
    sel = np.random.default_rng(cfg.split_seed).permutation(len(train_q))[:n]
    print(f"    [train_frac={frac}] using {n}/{len(train_q)} training queries")
    return np.sort(train_q[sel])


def artifact_batch(art, idx, n_patches, device):
    idx = np.asarray(idx, dtype=np.int64)
    counts = art["counts"][idx].float()
    return {
        "centroids": art["centers"][idx].to(device),
        "counts": art["counts"][idx].to(device),
        "count_feature": (torch.log1p(counts) / math.log1p(n_patches)).unsqueeze(-1).to(device),
        "labels": art["labels"][idx].to(device),
        "target_rank": art["target_rank"][idx].to(device),
        "slot_owner": art["slot_owner"][idx].to(device),
        "slot_rank": art["slot_rank"][idx].to(device),
        "slot_fraction": art["slot_fraction"][idx].to(device),
        "anchor_mask": art["anchor_mask"][idx].to(device),
    }


@torch.no_grad()
def teacher_scores(corpus, device, chunk=128):
    """Fine-vector (uncompressed) score matrix over all queries x all pages.

    Used ONLY for train-side hard-negative mining and train-query distillation targets, plus the
    reported ceiling. Never as a model input.
    """
    from .metrics import score_matrix
    return score_matrix(corpus.page, corpus.pmask, corpus.qemb, corpus.qmask,
                        np.arange(corpus.n_queries), device, d_chunk=chunk)


def mine_hard_negatives(tscores, corpus, train_q, pool_size):
    """For each TRAIN query, the hardest wrong TRAIN documents under the teacher."""
    train_docs = np.unique(corpus.gold[train_q].numpy())
    td = torch.as_tensor(train_docs, dtype=torch.long)
    pos_of = {int(d): r for r, d in enumerate(td)}
    out = {}
    for r, q in enumerate(train_q):
        v = tscores[int(q), td].clone()
        g = int(corpus.gold[int(q)])
        if g in pos_of:
            v[pos_of[g]] = -float("inf")             # never mine the gold page
        top = v.topk(min(pool_size, len(td))).indices
        out[int(q)] = td[top].tolist()
    return out


def candidate_docs(corpus, q_idx, neg_map, n_neg, rng):
    """[len(q_idx), 1+n_neg] doc indices: the gold page first, then mined hard negatives."""
    rows = []
    for q in q_idx:
        q = int(q)
        pool = neg_map[q]
        head = pool[:max(n_neg // 2, 1)]
        tail_n = n_neg - len(head)
        tail = rng.choice(pool[len(head):], size=min(tail_n, max(len(pool) - len(head), 1)),
                          replace=False).tolist() if tail_n > 0 and len(pool) > len(head) else []
        row = [int(corpus.gold[q]), *head, *tail]
        while len(row) < 1 + n_neg:
            row.append(row[-1])
        rows.append(row[:1 + n_neg])
    return np.asarray(rows, dtype=np.int64)


def training_step(model, corpus, art, q_idx, docs, weights, cfg, device, support_dirs):
    B, C = docs.shape
    flat = docs.reshape(-1)
    ab = artifact_batch(art, flat, corpus.n_patches, device)
    page = corpus.page[flat].to(device)
    pm = corpus.pmask[flat].to(device)

    refined, raw_norm = model.encode(ab["centroids"], page, pm)
    q = corpus.qemb[q_idx].to(device)
    qm = corpus.qmask[q_idx].to(device)
    pair_q = q.repeat_interleave(C, dim=0)
    pair_m = qm.repeat_interleave(C, dim=0)

    with torch.no_grad():
        t_tok = L.token_support(pair_q, page)
    c_tok = L.token_support(pair_q, refined)
    t_agg = L.aggregate(t_tok, pair_m).reshape(B, C)
    c_agg = L.aggregate(c_tok, pair_m).reshape(B, C)

    terms = {
        "code_token": L.masked_token_mse(c_tok, t_tok, pair_m),
        "code_listwise": L.listwise_kl(c_agg, t_agg, cfg.temperature),
    }

    if model.decoder is not None:
        gen = model.generate(refined, raw_norm, ab["count_feature"],
                             ab["slot_owner"], ab["slot_fraction"], ab["anchor_mask"])
        g_tok = L.token_support(pair_q, gen)
        g_agg = L.aggregate(g_tok, pair_m).reshape(B, C)
        pos = torch.arange(B, device=device) * C            # gold rows only for set losses
        terms.update({
            "gen_token": L.masked_token_mse(g_tok, t_tok, pair_m),
            "gen_listwise": L.listwise_kl(g_agg, t_agg, cfg.temperature),
            "overshoot": L.overshoot_penalty(g_agg, t_agg),
            "cluster_set": L.cluster_set_loss(
                gen[pos], page[pos], ab["slot_owner"][pos], ab["labels"][pos],
                ab["slot_rank"][pos], ab["target_rank"][pos], ab["counts"][pos],
                cfg.cluster_sample),
            "support": L.support_loss(gen[pos], page[pos], support_dirs),
        })

    loss = L.combine(terms, weights)
    diag = {"loss": float(loss.detach())}
    diag.update({k: float(v.detach()) for k, v in terms.items()})
    return loss, diag


@torch.no_grad()
def encode_corpus(model, corpus, art, device, batch=64):
    model.eval()
    ref, nrm = [], []
    for s in range(0, corpus.n_pages, batch):
        idx = np.arange(s, min(corpus.n_pages, s + batch))
        ab = artifact_batch(art, idx, corpus.n_patches, device)
        r, n = model.encode(ab["centroids"], corpus.page[idx].to(device),
                            corpus.pmask[idx].to(device))
        ref.append(r.cpu()); nrm.append(n.cpu())
    return torch.cat(ref).float(), torch.cat(nrm).float()


def make_decode_fn(model, corpus, art, refined_all, norm_all, device):
    """doc indices -> regenerated [n, P, D]. Query-free by construction."""
    @torch.no_grad()
    def fn(doc_idx):
        idx = np.asarray(doc_idx.cpu() if torch.is_tensor(doc_idx) else doc_idx, dtype=np.int64)
        ab = artifact_batch(art, idx, corpus.n_patches, device)
        return model.generate(refined_all[idx].to(device), norm_all[idx].to(device),
                              ab["count_feature"], ab["slot_owner"],
                              ab["slot_fraction"], ab["anchor_mask"])
    return fn


def train_one(corpus, art, k, cfg, device, out_dir):
    ckpt = os.path.join(out_dir, "checkpoints", f"{corpus.slug}_k{k}_{cfg.tag}.pt")
    if cfg.reuse and os.path.exists(ckpt):
        print(f"  [reuse] {corpus.slug} k={k}")
        return torch.load(ckpt, map_location="cpu")["result"]

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    train_q, hold_q = make_split(corpus.n_queries, cfg.split_seed, cfg.holdout_frac)
    train_q = subsample_train(train_q, cfg)
    weights = {"cluster_set": cfg.w_cluster_set, "support": cfg.w_support,
               "gen_token": cfg.w_gen_token, "gen_listwise": cfg.w_gen_listwise,
               "overshoot": cfg.w_overshoot, "code_token": cfg.w_code_token,
               "code_listwise": cfg.w_code_listwise}

    tsc = teacher_scores(corpus, device)
    neg_map = mine_hard_negatives(tsc, corpus, train_q, cfg.hard_negative_pool)
    support_dirs = L.make_support_dirs(cfg.support_dirs, corpus.dim).to(device)

    model = GLIE(corpus.dim, cfg.refiner, cfg.decoder, cfg.heads, cfg.dec_hidden,
                 cfg.refiner_alpha_max, cfg.refiner_gate_bias, cfg.decoder_alpha_max,
                 use_norm=not cfg.no_norm_channel,
                 use_count=not cfg.no_count_channel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)

    # TWO checkpoints from ONE run. glie_stage1 and glie_decoder are two deployable systems --
    # single-stage retrieval, and the two-stage cascade -- so each is selected on the metric it is
    # actually deployed under. Reading the stage-1 row off the decoder-best epoch understates it:
    # at high budgets the refiner overfits the train queries while the decoder keeps improving, so
    # the two peak at different epochs and one checkpoint reports a stage 1 that can fall BELOW the
    # baseline it refines. Both checkpoints are stored and both are disclosed.
    best, best_ep, best_state, bad, hist = -1.0, 0, None, 0, []
    best_s1, best_s1_ep, best_s1_state = -1.0, 0, None
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        rng = np.random.default_rng(cfg.seed * 10000 + ep)
        order = train_q.copy(); rng.shuffle(order)
        logs = []
        for s in range(0, len(order), cfg.query_batch):
            qi = order[s:s + cfg.query_batch]
            docs = candidate_docs(corpus, qi, neg_map, cfg.hard_negatives, rng)
            opt.zero_grad(set_to_none=True)
            loss, diag = training_step(model, corpus, art, qi, docs, weights, cfg,
                                       device, support_dirs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            logs.append(diag)
        sched.step()

        if ep == 1 or ep % cfg.eval_every == 0 or ep == cfg.epochs:
            refined_all, norm_all = encode_corpus(model, corpus, art, device)
            s1, _ = evaluate_static(refined_all, None, corpus, hold_q, device)
            if model.decoder is not None:
                dec = evaluate_cascade(refined_all, make_decode_fn(model, corpus, art,
                                                                   refined_all, norm_all, device),
                                       corpus, hold_q, device, cfg.topl, cfg.tail_policy)
            else:
                dec = s1
            score = dec["ndcg@5"]
            row = {"epoch": ep, "stage1_ndcg@5": s1["ndcg@5"], "decoder_ndcg@5": dec["ndcg@5"],
                   **{f"loss_{k2}": float(np.mean([g[k2] for g in logs if k2 in g]))
                      for k2 in logs[0]}}
            hist.append(row)
            if score > best + 1e-8:
                best, best_ep, bad = score, ep, 0
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            elif ep >= cfg.min_epochs:
                bad += 1
            if s1["ndcg@5"] > best_s1 + 1e-8:
                best_s1, best_s1_ep = s1["ndcg@5"], ep
                best_s1_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            print(f"  [{corpus.slug} k={k}] ep {ep:3d} loss={row['loss_loss']:.4f} "
                  f"stage1={s1['ndcg@5']:.4f} decoder={dec['ndcg@5']:.4f} "
                  f"best_dec={best:.4f}@{best_ep} best_s1={best_s1:.4f}@{best_s1_ep}")
            if cfg.patience and bad >= cfg.patience and ep >= cfg.min_epochs:
                print(f"  [{corpus.slug} k={k}] early stop at ep {ep}")
                break

    # stage-1 row: from the stage-1-best checkpoint (the single-stage deployment)
    model.load_state_dict(best_s1_state); model.eval()
    ref_s1, _ = encode_corpus(model, corpus, art, device)
    s1, _ = evaluate_static(ref_s1, None, corpus, hold_q, device)

    # decoder row: from the decoder-best checkpoint (the two-stage deployment)
    model.load_state_dict(best_state); model.eval()
    refined_all, norm_all = encode_corpus(model, corpus, art, device)
    s1_at_dec, _ = evaluate_static(refined_all, None, corpus, hold_q, device)
    dec_fn = make_decode_fn(model, corpus, art, refined_all, norm_all, device)
    dec = (evaluate_cascade(refined_all, dec_fn, corpus, hold_q, device, cfg.topl, cfg.tail_policy)
           if model.decoder is not None else s1_at_dec)
    orc = (evaluate_cascade(refined_all, dec_fn, corpus, hold_q, device, cfg.topl,
                            cfg.tail_policy, oracle=True) if model.decoder is not None else s1_at_dec)

    result = {
        "dataset": corpus.slug, "k": int(k), "tag": cfg.tag, "seed": cfg.seed,
        "refiner": cfg.refiner, "decoder": cfg.decoder,
        "stage1": s1, "decoder_metrics": dec, "oracle": orc,
        "stage1_at_decoder_ckpt": s1_at_dec,   # disclosed: what one shared checkpoint would give
        "best_epoch": best_ep, "best_stage1_epoch": best_s1_ep,
        "epochs_ran": hist[-1]["epoch"] if hist else 0,
        "minutes": round((time.time() - t0) / 60.0, 2),
        "trainable_params": model.n_params(),
        "storage": storage_bytes(k, corpus.dim, "glie"),
        "compression_vs_full": compression_vs_full(k, corpus.n_patches),
        "history": hist,
        "loss_weights": weights,
    }
    torch.save({"state_dict": best_state, "stage1_state_dict": best_s1_state,
                "result": result}, ckpt)
    return result


def train_multi(sources, k, cfg, device, out_dir, tag_suffix="multi"):
    """Fit ONE codec on several corpora at once, for cross-domain transfer.

    `sources` is a list of (corpus, artifact). Batches are interleaved across corpora within each
    epoch, so the codec sees several page styles per update rather than one domain per run.

    Model selection uses the POOLED HOLDOUT OF THE TRAINING CORPORA ONLY. The transfer target is
    never touched -- selecting on it would be exactly the leak this experiment exists to rule out.
    """
    ckpt = os.path.join(out_dir, "checkpoints", f"MULTI_{tag_suffix}_k{k}_{cfg.tag}.pt")
    if cfg.reuse and os.path.exists(ckpt):
        print(f"  [reuse] multi-source codec k={k}")
        return ckpt

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    weights = {"cluster_set": cfg.w_cluster_set, "support": cfg.w_support,
               "gen_token": cfg.w_gen_token, "gen_listwise": cfg.w_gen_listwise,
               "overshoot": cfg.w_overshoot, "code_token": cfg.w_code_token,
               "code_listwise": cfg.w_code_listwise}
    dim = sources[0][0].dim
    support_dirs = L.make_support_dirs(cfg.support_dirs, dim).to(device)

    prepared = []
    for corpus, art in sources:
        tr_q, ho_q = make_split(corpus.n_queries, cfg.split_seed, cfg.holdout_frac)
        tr_q = subsample_train(tr_q, cfg)
        tsc = teacher_scores(corpus, device)
        neg = mine_hard_negatives(tsc, corpus, tr_q, cfg.hard_negative_pool)
        prepared.append({"corpus": corpus, "art": art, "train_q": tr_q,
                         "hold_q": ho_q, "neg": neg})
        print(f"    source {corpus.slug}: {len(tr_q)} train / {len(ho_q)} holdout queries")

    model = GLIE(dim, cfg.refiner, cfg.decoder, cfg.heads, cfg.dec_hidden,
                 cfg.refiner_alpha_max, cfg.refiner_gate_bias, cfg.decoder_alpha_max,
                 use_norm=not cfg.no_norm_channel,
                 use_count=not cfg.no_count_channel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)

    best, best_ep, best_state, bad = -1.0, 0, None, 0
    best_s1, best_s1_state = -1.0, None
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        rng = np.random.default_rng(cfg.seed * 10000 + ep)
        # one shuffled stream of (source, query-batch) so updates alternate across domains
        batches = []
        for si, p in enumerate(prepared):
            order = p["train_q"].copy(); rng.shuffle(order)
            for s in range(0, len(order), cfg.query_batch):
                batches.append((si, order[s:s + cfg.query_batch]))
        rng.shuffle(batches)

        for si, qi in batches:
            p = prepared[si]
            docs = candidate_docs(p["corpus"], qi, p["neg"], cfg.hard_negatives, rng)
            opt.zero_grad(set_to_none=True)
            loss, _ = training_step(model, p["corpus"], p["art"], qi, docs, weights, cfg,
                                    device, support_dirs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        if ep == 1 or ep % cfg.eval_every == 0 or ep == cfg.epochs:
            s1s, decs = [], []
            for p in prepared:
                ref, nrm = encode_corpus(model, p["corpus"], p["art"], device)
                m1, _ = evaluate_static(ref, None, p["corpus"], p["hold_q"], device)
                s1s.append(m1["ndcg@5"])
                if model.decoder is not None:
                    fn = make_decode_fn(model, p["corpus"], p["art"], ref, nrm, device)
                    md = evaluate_cascade(ref, fn, p["corpus"], p["hold_q"], device,
                                          cfg.topl, cfg.tail_policy)
                    decs.append(md["ndcg@5"])
                else:
                    decs.append(m1["ndcg@5"])
            s1m, decm = float(np.mean(s1s)), float(np.mean(decs))
            if decm > best + 1e-8:
                best, best_ep, bad = decm, ep, 0
                best_state = {n: q.detach().cpu().clone() for n, q in model.state_dict().items()}
            elif ep >= cfg.min_epochs:
                bad += 1
            if s1m > best_s1 + 1e-8:
                best_s1 = s1m
                best_s1_state = {n: q.detach().cpu().clone() for n, q in model.state_dict().items()}
            print(f"  [multi k={k}] ep {ep:3d} src-holdout stage1={s1m:.4f} decoder={decm:.4f} "
                  f"best={best:.4f}@{best_ep}")
            if cfg.patience and bad >= cfg.patience and ep >= cfg.min_epochs:
                print(f"  [multi k={k}] early stop at ep {ep}")
                break

    torch.save({"state_dict": best_state, "stage1_state_dict": best_s1_state,
                "result": {"sources": [p["corpus"].slug for p in prepared], "k": int(k),
                           "best_epoch": best_ep, "src_holdout_decoder": best,
                           "minutes": round((time.time() - t0) / 60.0, 2)}}, ckpt)
    print(f"  saved multi-source codec k={k} "
          f"(sources: {', '.join(p['corpus'].slug for p in prepared)}, {best:.4f} @ep{best_ep})")
    return ckpt


@torch.no_grad()
def evaluate_transfer(ckpt_path, corpus, art, k, cfg, device):
    """Apply a codec fitted on ANOTHER corpus, frozen, to this one.

    The refiner and decoder consume only (centroids, page vectors, cluster layout) -- never a
    query and never anything corpus-specific -- so transfer is just: run this corpus's own k-means
    through the frozen modules. This is the zero-shot setting: no labelled queries are needed for
    the target corpus, only its pages.
    """
    saved = torch.load(ckpt_path, map_location="cpu")
    model = GLIE(corpus.dim, cfg.refiner, cfg.decoder, cfg.heads, cfg.dec_hidden,
                 cfg.refiner_alpha_max, cfg.refiner_gate_bias, cfg.decoder_alpha_max,
                 use_norm=not cfg.no_norm_channel,
                 use_count=not cfg.no_count_channel).to(device)
    _, hold_q = make_split(corpus.n_queries, cfg.split_seed, cfg.holdout_frac)

    model.load_state_dict(saved["stage1_state_dict"]); model.eval()
    ref_s1, _ = encode_corpus(model, corpus, art, device)
    s1, _ = evaluate_static(ref_s1, None, corpus, hold_q, device)

    model.load_state_dict(saved["state_dict"]); model.eval()
    refined_all, norm_all = encode_corpus(model, corpus, art, device)
    dec_fn = make_decode_fn(model, corpus, art, refined_all, norm_all, device)
    if model.decoder is not None:
        dec = evaluate_cascade(refined_all, dec_fn, corpus, hold_q, device, cfg.topl,
                               cfg.tail_policy)
        orc = evaluate_cascade(refined_all, dec_fn, corpus, hold_q, device, cfg.topl,
                               cfg.tail_policy, oracle=True)
    else:
        dec = orc = s1

    return {"dataset": corpus.slug, "k": int(k), "tag": cfg.tag, "transfer_from": cfg.train_on,
            "stage1": s1, "decoder_metrics": dec, "oracle": orc,
            "storage": storage_bytes(k, corpus.dim, "glie"),
            "compression_vs_full": compression_vs_full(k, corpus.n_patches),
            "minutes": 0.0, "best_epoch": -1, "trainable_params": model.n_params()}
