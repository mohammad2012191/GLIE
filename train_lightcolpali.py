"""Light-ColPali baseline (Ma et al., ACL 2025 Findings, arXiv:2506.04997).

Their method is semantic hierarchical clustering applied at the post-projector stage WITH the
backbone LoRA-fine-tuned around it. The merge alone is training-free and is available as the
`cluster_merge` baseline; this script adds the fine-tuning half.

Scored by the SAME harness as every other method (metrics.evaluate_static), on the SAME split.
Because the fine-tune changes the encoder, both pages and queries are re-encoded with the tuned
model before scoring -- the cached embeddings belong to the frozen backbone and must not be reused.

TWO DISCLOSURES THAT BELONG IN THE PAPER:

1. BATCH SIZE. The paper uses batch 256; in-batch-negative InfoNCE draws its negatives from a
   single forward pass, so a smaller batch is a genuinely weaker objective and --grad_accum does
   not fix it. Batch 256 of live-encoded pages does not fit one A100 (it needs cross-GPU embedding
   all-gather). This is therefore a single-GPU reproduction and a LOWER BOUND.

2. LEARNING RATE. The paper's 5e-4 is paired with batch 256. At batch 8 it is ~32x too hot per
   sample and destroys the backbone: a 42-hour run at 5e-4 scored 0.19-0.23 nDCG@5 at k=4, far
   BELOW the ~0.57 the same merge scores at LoRA init. Default here is scaled to the batch.

ALWAYS run --eval_at_init first. LoRA initializes to zero, so that number is stock ColPali + merge:
the training-free floor. A fine-tuned score below it means the optimizer is damaging the backbone,
and the fix is a lower learning rate, not more data.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

from .config import DEV_SUBSETS
from .data import Corpus, load_corpus, make_split
from .metrics import evaluate_static, storage_bytes


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEV_SUBSETS)
    p.add_argument("--train_dataset", default="vidore/colpali_train_set")
    p.add_argument("--train_split", default="train")
    p.add_argument("--model_name", default="vidore/colpali-v1.3")
    p.add_argument("--cache_dir", default="/ibex/user/hamidme/glie_cache")
    p.add_argument("--out_dir", default="/ibex/user/hamidme/glie_out")
    p.add_argument("--codes", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    p.add_argument("--max_train_pairs", type=int, default=4000,
                   help="MATCHED to GLIE's training budget. Also what makes this tractable: "
                        "uncapped it is ~49 GPU-hours per budget, ~10 days for a sweep.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="scaled for batch 8; the paper's 5e-4 assumes batch 256")
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--loss", choices=["pairwise", "engine_pairwise", "infonce"], default="pairwise",
                   help="pairwise (default) = ColbertPairwiseCELoss computed locally, respecting our "
                        "padding masks, on raw-sum scores. This is the loss ColPali itself is "
                        "trained with: softplus over the hardest negative, which SATURATES once the "
                        "margin is met instead of pushing forever. engine_pairwise = colpali_engine's "
                        "own class, for cross-checking (ignores our masks, normalizes internally). "
                        "infonce = CE over all in-batch negatives. The official cookbook fine-tunes "
                        "ColPali at batch 4 with the pairwise loss, so batch size was never the "
                        "blocker here -- the objective was.")
    p.add_argument("--eval_every_steps", type=int, default=200)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--eval_at_init", action="store_true",
                   help="score the merge at LoRA init (= stock ColPali + merge) and exit")
    p.add_argument("--eval_only", action="store_true",
                   help="skip training: load the saved adapter from <out_dir>/lcp_ckpt/k<k> and "
                        "evaluate. Use to recover results whose JSON was lost, or to re-score an "
                        "existing sweep on more subsets, without repeating the fine-tune.")
    p.add_argument("--encode_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="lightcolpali")
    return p.parse_args()


# ---------------------------------------------------------------------------
def differentiable_merge(emb, k):
    """(P,D) -> (k,D), gradient-carrying.

    The cluster ASSIGNMENT is a discrete decision and is computed detached; the merged vectors are
    means of the ORIGINAL grad-carrying embeddings, so gradients still reach the backbone.
    """
    n = emb.shape[0]
    kk = max(1, min(k, n))
    if kk >= n:
        return F.normalize(emb, dim=-1)
    with torch.no_grad():
        x = F.normalize(emb.float(), dim=-1).cpu().numpy().astype(np.float64)
        d = 1.0 - x @ x.T
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        lab = AgglomerativeClustering(n_clusters=kk, metric="precomputed",
                                      linkage="average").fit_predict(d)
    lab = torch.from_numpy(lab).to(emb.device)
    merged = torch.stack([emb[lab == j].mean(0) for j in range(kk) if (lab == j).any()])
    return F.normalize(merged, dim=-1)


def batch_merge(page, mask, k):
    out = [differentiable_merge(page[b][mask[b]], k) for b in range(page.shape[0])]
    m = max(o.shape[0] for o in out)
    pad = page.new_zeros(len(out), m, page.shape[-1])
    msk = torch.zeros(len(out), m, dtype=torch.bool, device=page.device)
    for b, o in enumerate(out):
        pad[b, :o.shape[0]] = o; msk[b, :o.shape[0]] = True
    return pad, msk


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_model(model_name, lora_r, lora_alpha, lora_dropout, device, adapter=None,
                verbose=True):
    """ColPali + a fresh LoRA, with ColPali's OWN adapter neutralized.

    THE TRAP THIS AVOIDS. `vidore/colpali-v1.3` is itself a PEFT checkpoint: it ships LoRA weights
    on the MLP projections and on `custom_text_proj` (the final 128-d projection). Wrapping it in a
    second LoRA and calling `get_peft_model` marks EVERY parameter whose name contains `lora_` as
    trainable -- including ColPali's own. Worse, `custom_text_proj`'s adapter fails to load from the
    checkpoint (a `base_model.model.` key-prefix mismatch) and is freshly initialized. PEFT
    zero-inits `lora_B`, so the model is healthy at step 0 and the failure is invisible in an
    eval-at-init; but as soon as the learning rate becomes nonzero, an untrained adapter starts
    rewriting the output embedding space and retrieval collapses. Observed: nDCG@5 0.54 -> 0.045
    at k=4, with the cliff landing exactly where warmup ends. No learning rate fixes this.

    Fix: merge and unload the pre-existing adapter where possible, then hard-freeze anything that is
    not one of OUR target modules, and report what is actually trainable.
    """
    from colpali_engine.models import ColPali, ColPaliProcessor
    from peft import LoraConfig, get_peft_model, PeftModel

    proc = ColPaliProcessor.from_pretrained(model_name)
    base = ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                   device_map=device, attn_implementation="sdpa")

    # 1. Bake ColPali's own adapter into the base weights and remove it from the graph.
    #    custom_text_proj's adapter has lora_B == 0, so merging it is a no-op -- exactly the
    #    behaviour we already validated at init -- while the correctly-loaded MLP adapters merge
    #    properly. Either way, nothing of ColPali's is left trainable afterwards.
    if hasattr(base, "merge_and_unload"):
        try:
            base = base.merge_and_unload()
            if verbose:
                print("  [build] merged and unloaded ColPali's pre-existing LoRA")
        except Exception as e:
            print(f"  [build] merge_and_unload failed ({e}); relying on the freeze below")

    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()

    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=TARGET_MODULES, task_type="FEATURE_EXTRACTION"))

    # 2. Hard freeze: only adapters on OUR target modules may train. This is the backstop that
    #    catches any pre-existing adapter merge_and_unload did not remove.
    frozen = 0
    for n, p in model.named_parameters():
        if p.requires_grad and not any(t in n for t in TARGET_MODULES):
            p.requires_grad_(False); frozen += 1

    if verbose:
        train_names = [n for n, p in model.named_parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [build] froze {frozen} non-target adapter tensors; "
              f"{len(train_names)} tensors / {n_train/1e6:.1f}M params trainable")
        stray = [n for n in train_names if "custom_text_proj" in n or "mlp." in n]
        assert not stray, (
            f"ColPali's own adapter is still trainable: {stray[:4]}. Training these -- especially "
            f"custom_text_proj, which loads randomly initialized -- destroys the embedding space.")
    return model, proc


@torch.no_grad()
def encode_with_model(model, proc, dataset, corpus, k, device, encode_batch=8):
    """Re-encode pages (merged to k) and queries with the CURRENT backbone.

    Returns a Corpus carrying tuned embeddings but the original gold mapping, so the shared eval
    harness scores this exactly like every other method.
    """
    from datasets import load_dataset
    model.eval()
    ds = load_dataset(dataset, split="test")
    rows = list(range(len(ds)))[:corpus.n_pages]

    merged_all, mask_all = [], []
    for s in range(0, len(rows), encode_batch):
        imgs = [ds[i]["image"].convert("RGB") for i in rows[s:s + encode_batch]]
        b = proc.process_images(imgs).to(device)
        out = model(**b)
        pm = b.get("attention_mask")
        pm = pm.bool() if pm is not None else torch.ones(out.shape[:2], dtype=torch.bool,
                                                         device=out.device)
        mg, mk = batch_merge(out.float(), pm, k)
        merged_all.append(mg.cpu()); mask_all.append(mk.cpu())

    q_rows = corpus.gold.tolist()
    queries = [ds[rows[r]]["query"] for r in q_rows]
    qe, qm = [], []
    for s in range(0, len(queries), encode_batch):
        b = proc.process_queries(queries[s:s + encode_batch]).to(device)
        out = model(**b)
        am = b["attention_mask"].bool().cpu()
        qe.extend(out[i].float().cpu() for i in range(out.shape[0]))
        qm.extend(am[i] for i in range(am.shape[0]))

    P = max(t.shape[1] for t in merged_all)
    N = sum(t.shape[0] for t in merged_all)
    D = merged_all[0].shape[-1]
    page = torch.zeros(N, P, D); pmask = torch.zeros(N, P, dtype=torch.bool)
    off = 0
    for t, m in zip(merged_all, mask_all):
        page[off:off + t.shape[0], :t.shape[1]] = t
        pmask[off:off + t.shape[0], :m.shape[1]] = m
        off += t.shape[0]

    Lq = max(t.shape[0] for t in qe)
    qemb = torch.zeros(len(qe), Lq, D); qmask = torch.zeros(len(qe), Lq, dtype=torch.bool)
    for i, (e, m) in enumerate(zip(qe, qm)):
        qemb[i, :e.shape[0]] = e; qmask[i, :m.shape[0]] = m
    qmask &= qemb.norm(dim=-1) > 1e-8

    model.train()
    return Corpus(corpus.slug, page, pmask, qemb, qmask, corpus.gold, corpus.meta)


def maxsim_sum(q, qm, docs, dm=None):
    """RAW-SUM MaxSim, matching colpali_engine's `_inbatch_scores`.

    Deliberately NOT divided by query length. Normalizing is harmless for eval ranking (it is a
    per-query constant) but it is destructive INSIDE the loss: it bounds the logits to roughly
    [0, 1], and cross-entropy then demands a logit gap of ~5 to drive p_positive toward 1, which
    the achievable score range cannot supply. The optimizer chases an unreachable target and wrecks
    the embedding geometry. This is what made an earlier version of this script collapse.
    """
    sim = torch.einsum("bqd,cpd->bcqp", q, docs)
    if dm is not None:
        sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
    return sim.amax(3).masked_fill(~qm[:, None, :], 0.0).sum(2)


def pairwise_ce_loss(q, qm, docs, dm, temperature=1.0):
    """`softplus((hardest_negative - positive) / T)`, i.e. colpali_engine's ColbertPairwiseCELoss.

    This is the loss ColPali itself is trained with. Unlike cross-entropy over all negatives it
    SATURATES: once the positive clears the hardest negative by a margin the gradient vanishes, so
    the optimizer stops pushing instead of distorting the representation to chase p=1.
    """
    scores = maxsim_sum(q, qm, docs, dm)
    pos = scores.diagonal()
    top2 = scores.topk(2, dim=1).values
    neg = torch.where(top2[:, 0] == pos, top2[:, 1], top2[:, 0])
    return F.softplus((neg - pos) / temperature).mean()


def infonce_loss(q, qm, docs, dm, temperature=1.0):
    """Cross-entropy over all in-batch negatives (colpali_engine's ColbertLoss), on RAW sums."""
    scores = maxsim_sum(q, qm, docs, dm) / temperature
    return F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))


def build_loss(name, temperature):
    """Select the training objective.

    `pairwise` (default) is our local ColbertPairwiseCELoss: same formula as colpali_engine's, but
    it RESPECTS OUR MASKS and uses the raw sum. The engine's version computes MaxSim internally from
    (query, doc) only, so it cannot see our padded query tokens or padded merged-doc slots, and it
    applies its own length normalization -- which is why its internal bounds check fires
    ("Scores out of bounds after normalization") on real embeddings, where a query token's best
    match can be slightly negative. Use `engine_pairwise` only to cross-check against the library.
    """
    if name == "engine_pairwise":
        from colpali_engine.loss import ColbertPairwiseCELoss
        fn = ColbertPairwiseCELoss()
        print("  [loss] colpali_engine.loss.ColbertPairwiseCELoss "
              "(NOTE: ignores our padding masks and normalizes internally)")
        return lambda q, qm, d, dm: fn(q, d)
    if name == "infonce":
        print("  [loss] cross-entropy over all in-batch negatives, raw-sum scores")
        return lambda q, qm, d, dm: infonce_loss(q, qm, d, dm, temperature)
    print("  [loss] local ColbertPairwiseCELoss (masked, raw-sum) -- the recommended default")
    return lambda q, qm, d, dm: pairwise_ce_loss(q, qm, d, dm, temperature)


def main():
    a = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "lcp_ckpt"), exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    corpora = {d: load_corpus(d, a.cache_dir, a.model_name, a.encode_batch, 0, device)
               for d in a.datasets}
    splits = {d: make_split(c.n_queries, a.split_seed, a.holdout_frac)
              for d, c in corpora.items()}

    # MERGE with any existing results for this tag instead of truncating them. Running budgets in
    # separate invocations under one tag (e.g. `--codes 4 8 16 32 64` then `--codes 2`) used to
    # silently erase everything from the first run.
    out_path = os.path.join(a.out_dir, "results", f"{a.tag}__results.json")
    rows = []
    if os.path.exists(out_path):
        try:
            rows = json.load(open(out_path))
            keep = {int(k) for k in a.codes}
            rows = [r for r in rows if int(r.get("k", -1)) not in keep]   # re-run budgets replace
            print(f"  [results] merging with {len(rows)} existing rows in {out_path}")
        except Exception as e:
            print(f"  [results] could not read existing {out_path} ({e}); starting fresh")
            rows = []
    for k in a.codes:
        ck = os.path.join(a.out_dir, "lcp_ckpt", f"k{k}")
        if a.eval_only:
            if not os.path.exists(ck):
                print(f"  [k={k}] no adapter at {ck}, skipping"); continue
            print(f"  [k={k}] eval-only: loading {ck}")
        model, proc = build_model(a.model_name, a.lora_r, a.lora_alpha, a.lora_dropout, device,
                                  adapter=ck if a.eval_only else None)

        if a.eval_at_init:
            print(f"\n=== merge-only floor at LoRA init, k={k} (no training) ===")
            for d, c in corpora.items():
                tuned = encode_with_model(model, proc, d, c, k, device, a.encode_batch)
                m, _ = evaluate_static(tuned.page, tuned.pmask, tuned, splits[d][1], device)
                rows.append({"dataset": c.slug, "k": k, "method": "cluster_merge_init", **m})
                print(f"  {c.slug:<40} nDCG@5={m['ndcg@5']:.4f}  nDCG@10={m['ndcg@10']:.4f}")
            del model
            torch.cuda.empty_cache()
            continue

        # ---- fine-tune (skipped in --eval_only) ------------------------------
        if a.eval_only:
            for d, c in corpora.items():
                tuned = encode_with_model(model, proc, d, c, k, device, a.encode_batch)
                m, _ = evaluate_static(tuned.page, tuned.pmask, tuned, splits[d][1], device)
                rows.append({"dataset": c.slug, "k": k, "method": "light_colpali", **m,
                             **storage_bytes(k, c.dim, "light_colpali")})
                print(f"  {c.slug:<40} k={k} nDCG@5={m['ndcg@5']:.4f}")
            del model
            torch.cuda.empty_cache()
            json.dump(rows, open(out_path, "w"), indent=2)
            continue

        from datasets import load_dataset
        n_fetch = int(a.max_train_pairs * 1.3) + 32
        train = load_dataset(a.train_dataset, split=f"{a.train_split}[:{n_fetch}]").shuffle(seed=a.seed)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        total = (a.max_train_pairs // a.batch) * a.epochs
        warm = max(1, int(a.warmup * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / warm))
        loss_fn = build_loss(a.loss, a.temperature)

        step = 0
        loss_hist = []
        for ep in range(a.epochs):
            imgs, qs, used = [], [], 0
            for row in train:
                if used >= a.max_train_pairs:
                    break
                q = row["query"]
                if q is None or not str(q).strip():
                    continue
                imgs.append(row["image"].convert("RGB")); qs.append(q)
                if len(imgs) < a.batch:
                    continue
                bi = proc.process_images(imgs).to(device)
                out = model(**bi)
                pm = bi.get("attention_mask")
                pm = pm.bool() if pm is not None else torch.ones(out.shape[:2], dtype=torch.bool,
                                                                 device=out.device)
                mg, mk = batch_merge(out, pm, k)
                bq = proc.process_queries(qs).to(device)
                qo = model(**bq)
                raw = loss_fn(qo, bq["attention_mask"].bool(), mg, mk)
                loss_hist.append(float(raw.detach()))
                loss = raw / a.grad_accum
                loss.backward()
                if (step + 1) % a.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
                step += 1; used += a.batch; imgs, qs = [], []
                if step % a.eval_every_steps == 0:
                    d0 = a.datasets[0]
                    tuned = encode_with_model(model, proc, d0, corpora[d0], k, device, a.encode_batch)
                    m, _ = evaluate_static(tuned.page, tuned.pmask, tuned, splits[d0][1], device)
                    # Print the LOSS alongside the metric. Loss falling while nDCG falls is the
                    # signature of an objective that is satisfiable in a way that ruins retrieval;
                    # both falling together means the optimizer is simply diverging.
                    recent = float(np.mean(loss_hist[-a.eval_every_steps:])) if loss_hist else float("nan")
                    print(f"  [k={k}] step {step} loss={recent:.4f} "
                          f"{corpora[d0].slug} nDCG@5={m['ndcg@5']:.4f}")
            print(f"  [k={k}] epoch {ep + 1}/{a.epochs} done ({used} pairs)")

        model.save_pretrained(os.path.join(a.out_dir, "lcp_ckpt", f"k{k}"))
        for d, c in corpora.items():
            tuned = encode_with_model(model, proc, d, c, k, device, a.encode_batch)
            m, _ = evaluate_static(tuned.page, tuned.pmask, tuned, splits[d][1], device)
            rows.append({"dataset": c.slug, "k": k, "method": "light_colpali", **m,
                         **storage_bytes(k, c.dim, "light_colpali")})
            print(f"  {c.slug:<40} k={k} nDCG@5={m['ndcg@5']:.4f}")
        del model
        torch.cuda.empty_cache()
        json.dump(rows, open(out_path, "w"), indent=2)

    json.dump(rows, open(os.path.join(a.out_dir, "results", f"{a.tag}__results.json"), "w"),
              indent=2)
    print(f"\nwritten to {os.path.join(a.out_dir, 'results', a.tag + '__results.json')}")


if __name__ == "__main__":
    main()
