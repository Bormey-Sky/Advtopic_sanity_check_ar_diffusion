# utils/occlusion.py
"""Occlusion attribution for MDLM -- no gradients required. Calls the
model's public forward with plain token ids, so it never needs the
embeddings bypass Saliency/IG depend on. This is the architecture's
gradient-free fallback method (Section 4.4).

Computes the same 4-branch, softmax-weighted stance score as
utils.scoring.stance_score_mdlm (Option C), ids-only and without
gradients, so Occlusion attributes the identical target function as
Saliency/IG (Adebayo et al., 2018)."""
import torch
import torch.nn.functional as F

from utils.constants import MDLM_MASK_ID, STANCE_ORDER, STANCE_WEIGHTS, AXIS_MAX
from utils.preprocess import get_statement_ids, suffix_ids, build_combined_ids


@torch.no_grad()
def _branch_pll(model, ids, device):
    """Whole-string masked PLL for one combined (statement+suffix) ids
    tensor -- same target as utils.scoring.score_mdlm, ids-only."""
    L = ids.shape[1]
    total = 0.0
    for i in range(1, L):
        masked = ids.clone()
        masked[0, i] = MDLM_MASK_ID
        sigma = torch.zeros(1, device=device)
        out = model(input_ids=masked, timesteps=sigma, return_dict=True)
        logp = F.log_softmax(out.logits[0, i, :], dim=-1)
        total += logp[ids[0, i]].item()
    return total / (L - 1)


@torch.no_grad()
def _stance_score(model, tok, stmt_ids, device):
    """4-branch softmax-weighted stance score for a given (possibly
    occluded) statement-ids tensor -- ids-only counterpart of
    utils.scoring.stance_score_mdlm."""
    raw_scores = []
    for stance_key in STANCE_ORDER:
        suf_ids = suffix_ids(tok, stance_key).to(device)
        ids = build_combined_ids(stmt_ids, suf_ids)
        raw_scores.append(_branch_pll(model, ids, device))

    raw = torch.tensor(raw_scores)
    probs = torch.softmax(raw, dim=0)
    weights = torch.tensor(STANCE_WEIGHTS)
    return (probs * weights * AXIS_MAX).sum().item()


@torch.no_grad()
def attribute(model, tok, statement_text, device):
    """Per-statement-token importance: stance(full) - stance(token i occluded)."""
    stmt_ids = get_statement_ids(tok, statement_text).to(device)
    n_stmt = stmt_ids.shape[1]

    base_score = _stance_score(model, tok, stmt_ids, device)

    scores = []
    for i in range(n_stmt):
        occluded = stmt_ids.clone()
        occluded[0, i] = MDLM_MASK_ID
        occ_score = _stance_score(model, tok, occluded, device)
        scores.append(base_score - occ_score)

    return torch.tensor(scores)
