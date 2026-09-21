# utils/scoring.py
"""MDLM-only scoring. Whole-string masked PLL (Salazar et al., 2020:
mask one position at a time, average log-prob of the true token),
differentiable through the embeds bypass so Saliency/IG can backprop
into it.

stance_score_from_embeds() / stance_score_mdlm() are the actual target
function attribution runs against: Option C, the poster's full
4-branch softmax-weighted combine (score_statement() in the poster's
evaluation_v2.py), reimplemented here as a differentiable whole-string
PLL per branch so gradients can flow through it -- the poster's own
compute_pll() runs under torch.no_grad() and can't be reused directly
for attribution.

stance_score_from_embeds() takes the statement embeddings as a plain
argument rather than computing them internally, so Integrated
Gradients can call it at interpolated points along the path;
stance_score_mdlm() is the thin wrapper Saliency uses when it just
wants the real embeddings with grad tracking."""
import torch
import torch.nn.functional as F

from utils.constants import STANCE_ORDER, STANCE_WEIGHTS, AXIS_MAX, MDLM_MASK_ID
from utils.preprocess import (
    get_statement_ids,
    suffix_ids,
    get_embeddings,
    build_combined_ids,
    build_combined_embeds,
)


def dit_backbone_forward_from_embeds(backbone, x, sigma):
    """Mirrors DITBackbone.forward, takes embeddings directly since
    inputs_embeds has no public parameter. Validated on A100."""
    if not backbone.config.time_conditioning:
        sigma = torch.zeros_like(sigma)

    c = F.silu(backbone.sigma_map(sigma))
    rotary_cos_sin = backbone.rotary_emb(c)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for block in backbone.blocks:
            x = block(x, rotary_cos_sin, c, seqlens=None)
        logits = backbone.output_layer(x, c)

    return logits, None


def score_mdlm(model, embeds, ids, mask_embedding, device):
    """Whole-string masked PLL for ONE stance branch, differentiable.
    Averages log-prob of the true token over every position (statement
    + suffix), masking one position at a time. Feeds into the 4-branch
    combine below -- not a target on its own."""
    L = ids.shape[1]
    total = 0.0

    for i in range(1, L):
        masked_embeds = embeds.clone()
        masked_embeds[:, i, :] = mask_embedding

        sigma = torch.zeros(1, device=device)
        logits, _ = dit_backbone_forward_from_embeds(model.backbone, masked_embeds, sigma)

        logp = F.log_softmax(logits[0, i, :], dim=-1)
        total = total + logp[ids[0, i]]

    return total / (L - 1)


def get_mask_embedding(model, mdlm_mask_id, device):
    ids = torch.tensor([[mdlm_mask_id]], device=device)
    return get_embeddings(model, "mdlm_169m", ids)[0, 0, :]


def stance_score_from_embeds(model, tok, stmt_ids, stmt_embeds, device, mask_embedding=None):
    """Differentiable 4-branch stance score (Option C) for a given
    statement-ids tensor and a caller-supplied statement-embeddings
    tensor -- real (from stance_score_mdlm) or interpolated (from
    Integrated Gradients). Matches the poster's score_statement():
    whole-string masked PLL per stance branch, softmax-weighted combine
    (STANCE_ORDER / STANCE_WEIGHTS / AXIS_MAX, utils.constants)."""
    if mask_embedding is None:
        mask_embedding = get_mask_embedding(model, MDLM_MASK_ID, device)

    branch_scores = []
    for stance_key in STANCE_ORDER:
        suf_ids_t = suffix_ids(tok, stance_key).to(device)
        suf_embeds = get_embeddings(model, "mdlm_169m", suf_ids_t)

        ids = build_combined_ids(stmt_ids, suf_ids_t)
        embeds = build_combined_embeds(stmt_embeds, suf_embeds)

        branch_scores.append(score_mdlm(model, embeds, ids, mask_embedding, device))

    raw = torch.stack(branch_scores)
    probs = torch.softmax(raw, dim=0)
    weights = torch.tensor(STANCE_WEIGHTS, device=device, dtype=raw.dtype)
    return (probs * weights * AXIS_MAX).sum()


def stance_score_mdlm(model, tok, statement_text, device, mask_embedding=None):
    """Differentiable 4-branch stance score, starting from the real
    (non-interpolated) statement embeddings.

    Returns (stance, stmt_embeds): `stance` is a differentiable scalar
    tensor; `stmt_embeds` is the statement-token embedding tensor with
    requires_grad_ set, so callers doing attribution call
    stance.backward() and then read stmt_embeds.grad.

    Pass a precomputed `mask_embedding` (get_mask_embedding) when
    scoring many statements against the same checkpoint, to avoid
    recomputing it on every call.
    """
    if mask_embedding is None:
        mask_embedding = get_mask_embedding(model, MDLM_MASK_ID, device)

    stmt_ids = get_statement_ids(tok, statement_text).to(device)
    stmt_embeds = get_embeddings(model, "mdlm_169m", stmt_ids).detach().clone()
    stmt_embeds.requires_grad_(True)

    stance = stance_score_from_embeds(model, tok, stmt_ids, stmt_embeds, device, mask_embedding)
    return stance, stmt_embeds
