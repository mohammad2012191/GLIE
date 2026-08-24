"""GLIE models: the polar-decomposition refiner and the anchored generative decoder.

The organizing principle, and the reason the two stages are shaped the way they are:

    DIRECTION says where on the sphere the code sits. Retrieval reads only this.
    MAGNITUDE says how wide that patch of manifold is. Only regeneration reads this.

The split is exact, not heuristic. For unit vectors x_1..x_n with centroid c = mean(x_i):

    (1/n) sum_i ||x_i - c||^2 = 1 - ||c||^2

so the centroid norm is a sufficient statistic for the cluster's angular spread, and the k-means
objective on the sphere is sum_j n_j (1 - ||c_j||^2). The norms ARE the k-means objective:
normalizing the centroid and throwing the norm away discards the exact record of how well each
cluster was covered. We keep it as a per-cluster scalar and spend it at generation time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2n(x, eps=1e-8):
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def tangent(delta, anchor):
    """Component of `delta` orthogonal to `anchor`.

    A radial component is deleted by the subsequent renormalization, so letting the network spend
    capacity on it wastes parameters and conditions the optimization badly. Projecting it out means
    every unit of update produces actual angular movement.
    """
    return delta - anchor * (delta * anchor).sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Refiners. Both return (unit directions [B,k,D], raw norms [B,k,1]).
# ---------------------------------------------------------------------------
class GatedTangentRefiner(nn.Module):
    """Bounded tangent-space update with a dispersion-conditioned gate.

    The gate sees the cluster's own magnitude r and dispersion 1-r^2, so the model learns HOW FAR
    to move each centroid from how diffuse its cluster is: a tight cluster is already well
    represented and needs almost nothing, a diffuse one needs the correction. The gate bias starts
    at -4, so beta ~ 0.009 and the module begins essentially AT normalized k-means -- the same
    baseline-preserving property as a zero-initialized residual, but adaptive.
    """

    def __init__(self, dim, heads=4, alpha_max=0.5, gate_bias=-4.0):
        super().__init__()
        self.alpha_max = alpha_max
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.delta = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim + 2, 1)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, centroids, page, pmask=None):
        raw_norm = centroids.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = centroids / raw_norm
        dispersion = (1.0 - raw_norm.square()).clamp_min(0.0)   # = within-cluster variance
        kpm = ~pmask if pmask is not None else None
        upd, _ = self.attn(u, page, page, key_padding_mask=kpm, need_weights=False)
        t = tangent(self.delta(upd), u)
        t = t / (1.0 + t.norm(dim=-1, keepdim=True))            # bounded step
        beta = self.alpha_max * torch.sigmoid(self.gate(torch.cat([upd, raw_norm, dispersion], -1)))
        return l2n(u + beta * t), raw_norm


class ZeroInitRefiner(nn.Module):
    """Simpler ablation arm: additive zero-initialized correction, renormalized.

    Starts exactly at normalized k-means because `delta` is zero-initialized. Kept so the paper can
    report what the tangent projection and the gate are actually worth.
    """

    def __init__(self, dim, heads=4, **_):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward(self, centroids, page, pmask=None):
        raw_norm = centroids.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = centroids / raw_norm
        kpm = ~pmask if pmask is not None else None
        upd, _ = self.attn(u, page, page, key_padding_mask=kpm, need_weights=False)
        return l2n(u + self.delta(upd)), raw_norm


class PassthroughRefiner(nn.Module):
    """Normalized k-means, no learning. The training-free arm of the refiner ablation."""

    def __init__(self, dim, **_):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, centroids, page, pmask=None):
        raw_norm = centroids.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return centroids / raw_norm, raw_norm


REFINERS = {"gated_tangent": GatedTangentRefiner,
            "zero_init": ZeroInitRefiner,
            "none": PassthroughRefiner}


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def child_fourier(fraction):
    """Deterministic identity for the l-th child of a cluster, from its position l/n_j in [0,1].

    Parameter-free: no per-slot weights to learn, and it generalizes across cluster sizes.
    """
    feats = [fraction]
    for f in (1.0, 2.0, 4.0, 8.0):
        feats.extend([torch.sin(math.pi * f * fraction), torch.cos(math.pi * f * fraction)])
    return torch.stack(feats, dim=-1)                            # [..., 9]


class AnchoredClusterDecoder(nn.Module):
    """Expand k stored clusters back into P unit vectors.

    Three properties that matter more than capacity:

    1. COUNT-PROPORTIONAL SLOTS. Cluster j owns n_j output slots, so the decoder reproduces the
       real cluster structure of the page instead of a fixed learned position bank.
    2. EXACT ANCHOR. Slot 0 of each cluster emits the refined direction verbatim, so the decoded
       set CONTAINS the code set. MaxSim is a max over the set, hence
           MaxSim(q, decoded) >= MaxSim(q, code)   for every query token,
       i.e. the decoder can only add evidence, never destroy it. No init-temperature tuning is
       needed to preserve the stage-1 baseline: it is structural.
    3. BOUNDED TANGENT DISPLACEMENT. Children leave their anchor only along the sphere, by at most
       alpha_max, with the step conditioned on the cluster's stored magnitude and size -- which is
       precisely the spread information the norm encodes.
    """

    def __init__(self, dim, hidden=256, alpha_max=0.75, use_norm=True, use_count=True):
        super().__init__()
        self.alpha_max = alpha_max
        self.use_norm, self.use_count = use_norm, use_count
        self.code_proj = nn.Linear(dim, hidden)
        self.side_proj = nn.Linear(2, hidden)          # (raw norm, log-count) -- the spread channel
        self.child_proj = nn.Linear(9, hidden)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.delta = nn.Linear(hidden, dim)
        self.alpha = nn.Linear(hidden, 1)

    def forward(self, refined, raw_norm, count_feature, slot_owner, slot_fraction, anchor_mask):
        D = refined.shape[-1]
        # Ablation switches: zeroing a channel keeps the parameter count and the architecture
        # identical, so any change in the result is attributable to the INFORMATION, not capacity.
        if not self.use_norm:
            raw_norm = torch.zeros_like(raw_norm)
        if not self.use_count:
            count_feature = torch.zeros_like(count_feature)
        cluster_h = self.code_proj(refined) + self.side_proj(torch.cat([raw_norm, count_feature], -1))
        slot_h = torch.gather(cluster_h, 1, slot_owner.unsqueeze(-1).expand(-1, -1, cluster_h.shape[-1]))
        slot_h = slot_h + self.child_proj(child_fourier(slot_fraction))
        slot_h = self.fusion(slot_h)

        anchors = torch.gather(refined, 1, slot_owner.unsqueeze(-1).expand(-1, -1, D))
        t = l2n(tangent(self.delta(slot_h), anchors))
        alpha = self.alpha_max * torch.sigmoid(self.alpha(slot_h))
        generated = l2n(anchors + alpha * t)
        return torch.where(anchor_mask.unsqueeze(-1), anchors, generated)


class GLIE(nn.Module):
    """Refiner + decoder. `encode` produces what is STORED; `generate` is query-free."""

    def __init__(self, dim, refiner="gated_tangent", decoder="anchored", heads=4, hidden=256,
                 refiner_alpha_max=0.5, refiner_gate_bias=-4.0, decoder_alpha_max=0.75,
                 use_norm=True, use_count=True):
        super().__init__()
        self.refiner = REFINERS[refiner](dim, heads=heads, alpha_max=refiner_alpha_max,
                                         gate_bias=refiner_gate_bias)
        self.decoder = (AnchoredClusterDecoder(dim, hidden, decoder_alpha_max,
                                               use_norm=use_norm, use_count=use_count)
                        if decoder == "anchored" else None)

    def encode(self, centroids, page, pmask=None):
        return self.refiner(centroids, page, pmask)

    def generate(self, refined, raw_norm, count_feature, slot_owner, slot_fraction, anchor_mask):
        if self.decoder is None:
            raise RuntimeError("decoder disabled (--decoder none)")
        return self.decoder(refined, raw_norm, count_feature, slot_owner, slot_fraction, anchor_mask)

    def n_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
