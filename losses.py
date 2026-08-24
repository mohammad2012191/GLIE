"""Training objectives.

Seven terms, each behind its own weight so a leave-one-out ablation is a command-line change and
not a code change. They fall into three groups:

  OPERATOR-MATCHED (the ones that made this work at all)
    gen_token / code_token       per-query-token MaxSim distillation: match the value the operator
                                 actually reads, not average reconstruction error
    gen_listwise / code_listwise listwise KL against the fine-vector teacher over the candidate set
    overshoot                    one-sided penalty on negatives scoring ABOVE the teacher, which is
                                 the failure that destroys rerank precision

  SET-GEOMETRY (what the decoder needs and the code does not)
    cluster_set                  order-free Chamfer between generated children and real patches,
                                 computed WITHIN each cluster
    support                      support-function match on fixed directions: protects the extreme
                                 points of the convex hull, which is what MaxSim reads

Reconstruction MSE is deliberately absent: it optimizes a quantity anti-correlated with retrieval.
"""
import torch
import torch.nn.functional as F


def make_support_dirs(count, dim, seed=314159):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(count, dim, generator=g)
    return F.normalize(z, dim=-1)


def token_support(query, docs):
    """Per-query-token max over a document's vectors, paired 1:1 (not all-pairs)."""
    return torch.einsum("bqd,bpd->bqp", query, docs).amax(dim=2)


def masked_token_mse(student, teacher, mask):
    w = mask.float()
    return (((student - teacher) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def aggregate(token_scores, mask):
    return (token_scores * mask.float()).sum(1) / mask.sum(1).clamp_min(1.0)


def listwise_kl(student, teacher, temperature=0.07):
    t = F.softmax(teacher / temperature, dim=1)
    s = F.log_softmax(student / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * temperature ** 2


def overshoot_penalty(student, teacher):
    """Negatives (columns 1..) are penalized only for scoring ABOVE the teacher."""
    return F.relu(student[:, 1:] - teacher[:, 1:]).pow(2).mean()


def support_loss(pred, target, dirs):
    ps = torch.einsum("bpd,md->bpm", pred, dirs).amax(1)
    ts = torch.einsum("bpd,md->bpm", target, dirs).amax(1)
    return F.smooth_l1_loss(ps, ts)


def _group_by_cluster(vectors, owner, rank, counts, sample):
    """[B,P,D] -> [B,k,sample,D] plus a validity mask, grouping vectors by their cluster."""
    B, P, D = vectors.shape
    k = counts.shape[1]
    m = max(1, int(counts.max().item()))
    bidx = torch.arange(B, device=vectors.device)[:, None].expand(B, P)
    flat = ((bidx * k + owner) * m + rank.clamp(max=m - 1)).reshape(-1)
    pad = vectors.new_zeros(B * k * m, D).index_copy(0, flat, vectors.reshape(-1, D))
    pad = pad.reshape(B, k, m, D)

    valid = torch.arange(m, device=vectors.device)[None, None, :] < counts[:, :, None]
    key = torch.rand(B, k, m, device=vectors.device).masked_fill(~valid, float("inf"))
    take = min(int(sample), m)
    val, idx = key.topk(take, dim=2, largest=False)
    got = torch.gather(pad, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
    return got, torch.isfinite(val)


def cluster_set_loss(pred, target, pred_owner, target_labels, pred_rank, target_rank,
                     counts, sample=48):
    """Symmetric Chamfer between generated and real vectors, WITHIN each cluster.

    Order-free (the slot ordering inside a cluster is arbitrary), and local: a generated child is
    only ever asked to look like a patch of its own cluster, which is the structure the
    count-proportional decoder actually produces.
    """
    pg, pv = _group_by_cluster(pred, pred_owner, pred_rank, counts, sample)
    tg, tv = _group_by_cluster(target, target_labels, target_rank, counts, sample)
    sim = torch.einsum("bkid,bkjd->bkij", pg, tg)
    p_best = sim.masked_fill(~tv[:, :, None, :], -1e4).amax(3)
    t_best = sim.masked_fill(~pv[:, :, :, None], -1e4).amax(2)
    p_mean = p_best.masked_fill(~pv, 0.0).sum(2) / pv.sum(2).clamp_min(1)
    t_mean = t_best.masked_fill(~tv, 0.0).sum(2) / tv.sum(2).clamp_min(1)
    return (1.0 - 0.5 * (p_mean + t_mean)).mean()


def combine(terms, weights):
    """Weighted sum over ACTIVE terms only, so a zero weight costs nothing to compute."""
    total = None
    for name, value in terms.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        total = w * value if total is None else total + w * value
    return total
