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
_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23)


def van_der_corput(index, base, iters=20):
    """Low-discrepancy coordinate in [0,1) for an integer index. Deterministic, parameter-free."""
    x = index.float()
    f = torch.ones_like(x)
    out = torch.zeros_like(x)
    for _ in range(iters):
        f = f / base
        out = out + f * (x % base)
        x = torch.div(x, base, rounding_mode="floor")
    return out


def slot_coords(fraction, slot_rank, dims):
    """Identity of the l-th child of a cluster, as `dims` numbers in [0,1].

    dims=1 is the published form: the slot's position l/n_j. dims>1 appends low-discrepancy
    coordinates from the slot's integer rank; parameter-free and independent of cluster size.
    """
    if dims <= 1:
        return fraction.unsqueeze(-1)
    extra = [van_der_corput(slot_rank, b) for b in _HALTON_BASES[:dims - 1]]
    return torch.stack([fraction] + extra, dim=-1)


def child_fourier(coords):
    """[..., r] in [0,1] -> [..., 9r]. Identical to the old scalar form at r=1."""
    feats = [coords]
    for f in (1.0, 2.0, 4.0, 8.0):
        feats.extend([torch.sin(math.pi * f * coords), torch.cos(math.pi * f * coords)])
    return torch.cat(feats, dim=-1)


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

    def __init__(self, dim, hidden=256, alpha_max=0.75, use_norm=True, use_count=True,
                 rank=0, layers=1, child_dims=1):
        super().__init__()
        self.alpha_max = alpha_max
        self.use_norm, self.use_count = use_norm, use_count
        self.rank = int(rank)
        self.layers = max(1, int(layers))
        self.child_dims = max(1, int(child_dims))
        self.code_proj = nn.Linear(dim, hidden)
        self.side_proj = nn.Linear(2, hidden)          # (raw norm, log-count) -- the spread channel
        self.child_proj = nn.Linear(9 * self.child_dims, hidden)
        # DEPTH. One block is LayerNorm -> expand -> GELU -> contract -> GELU. layers=1 is the
        # published configuration and reproduces it exactly, so a depth sweep compares against the
        # main table rather than against a new architecture. Blocks are plain rather than residual
        # on purpose: a skip connection would change layers=1 too, and then the sweep would measure
        # depth and the skip together.
        blocks = []
        for _ in range(self.layers):
            blocks += [nn.LayerNorm(hidden),
                       nn.Linear(hidden, hidden * 2), nn.GELU(),
                       nn.Linear(hidden * 2, hidden), nn.GELU()]
        self.fusion = nn.Sequential(*blocks)
        if self.rank > 0:
            # LOW-DIMENSIONAL DISPLACEMENT. The paper measures the token cloud as a manifold of
            # intrinsic dimension about five, so at any anchor the children should move inside a
            # rank-r subspace of the tangent space, not all D-1 directions. `basis` predicts that
            # local frame from the CLUSTER (it varies over the surface), `coeff` predicts the r
            # coordinates from the SLOT. Nothing extra is stored: the frame is a function of the
            # code, so this constrains the decoder without touching the footprint.
            self.basis = nn.Linear(hidden, self.rank * dim)
            self.coeff = nn.Linear(hidden, self.rank)
        else:
            self.delta = nn.Linear(hidden, dim)        # unconstrained, full tangent space
        self.alpha = nn.Linear(hidden, 1)

    def forward(self, refined, raw_norm, count_feature, slot_owner, slot_fraction, anchor_mask,
                slot_rank=None):
        D = refined.shape[-1]
        if self.child_dims > 1 and slot_rank is None:
            raise RuntimeError("child_dims>1 needs slot_rank: pass artifact_batch(...)['slot_rank']")
        # Ablation switches: zeroing a channel keeps the parameter count and the architecture
        # identical, so any change in the result is attributable to the INFORMATION, not capacity.
        if not self.use_norm:
            raw_norm = torch.zeros_like(raw_norm)
        if not self.use_count:
            count_feature = torch.zeros_like(count_feature)
        cluster_h = self.code_proj(refined) + self.side_proj(torch.cat([raw_norm, count_feature], -1))
        slot_h = torch.gather(cluster_h, 1, slot_owner.unsqueeze(-1).expand(-1, -1, cluster_h.shape[-1]))
        slot_h = slot_h + self.child_proj(child_fourier(
            slot_coords(slot_fraction, slot_rank, self.child_dims)))
        slot_h = self.fusion(slot_h)

        anchors = torch.gather(refined, 1, slot_owner.unsqueeze(-1).expand(-1, -1, D))
        if self.rank > 0:
            frame = self.basis(cluster_h).view(*cluster_h.shape[:2], self.rank, D)
            frame = torch.gather(frame, 1, slot_owner[..., None, None].expand(-1, -1, self.rank, D))
            disp = torch.einsum("bsr,bsrd->bsd", self.coeff(slot_h), frame)
        else:
            disp = self.delta(slot_h)
        t = l2n(tangent(disp, anchors))
        alpha = self.alpha_max * torch.sigmoid(self.alpha(slot_h))
        generated = l2n(anchors + alpha * t)
        return torch.where(anchor_mask.unsqueeze(-1), anchors, generated)


class GLIE(nn.Module):
    """Refiner + decoder. `encode` produces what is STORED; `generate` is query-free."""

    def __init__(self, dim, refiner="gated_tangent", decoder="anchored", heads=4, hidden=256,
                 refiner_alpha_max=0.5, refiner_gate_bias=-4.0, decoder_alpha_max=0.75,
                 use_norm=True, use_count=True, decoder_rank=0, decoder_layers=1,
                 decoder_child_dims=1):
        super().__init__()
        self.refiner = REFINERS[refiner](dim, heads=heads, alpha_max=refiner_alpha_max,
                                         gate_bias=refiner_gate_bias)
        self.decoder = (AnchoredClusterDecoder(dim, hidden, decoder_alpha_max,
                                               use_norm=use_norm, use_count=use_count,
                                               rank=decoder_rank, layers=decoder_layers,
                                               child_dims=decoder_child_dims)
                        if decoder == "anchored" else None)

    def encode(self, centroids, page, pmask=None):
        return self.refiner(centroids, page, pmask)

    def generate(self, refined, raw_norm, count_feature, slot_owner, slot_fraction, anchor_mask,
                 slot_rank=None):
        if self.decoder is None:
            raise RuntimeError("decoder disabled (--decoder none)")
        return self.decoder(refined, raw_norm, count_feature, slot_owner, slot_fraction,
                            anchor_mask, slot_rank)

    def n_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def glie_from_state_dict(dim, state_dict, refiner="zero_init", **kw):
    """Rebuild a GLIE whose decoder SHAPE matches a saved checkpoint.

    The evaluation scripts used to hardcode hidden=256, which is right for every published run and
    silently wrong the moment one sweeps --dec_hidden or --dec_layers. Width, depth and rank are all
    recoverable from the saved tensors, so read them off the checkpoint instead of a constant.
    """
    if not any(key.startswith("decoder.") for key in state_dict):
        return GLIE(dim, refiner, "none", **kw)
    hidden = state_dict["decoder.code_proj.weight"].shape[0]
    layers = sum(1 for key, v in state_dict.items()
                 if key.startswith("decoder.fusion.") and key.endswith(".weight") and v.dim() == 1)
    rank = (state_dict["decoder.coeff.weight"].shape[0]
            if "decoder.coeff.weight" in state_dict else 0)
    child_dims = state_dict["decoder.child_proj.weight"].shape[1] // 9
    return GLIE(dim, refiner, "anchored", hidden=hidden, decoder_rank=rank,
                decoder_layers=max(1, layers), decoder_child_dims=max(1, child_dims), **kw)
