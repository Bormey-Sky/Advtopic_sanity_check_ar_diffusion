# utils/integrated_gradients.py
"""Integrated Gradients attribution for MDLM (Sundararajan et al.,
2017). Interpolates statement-token embeddings between a mask-token
baseline and the real embeddings, accumulating gradients of the same
4-branch stance score (utils.scoring.stance_score_from_embeds) that
Saliency and Occlusion attribute, so all three methods explain the
same target function."""
import torch

from utils.constants import IG_STEPS, MDLM_MASK_ID
from utils.preprocess import get_statement_ids, get_embeddings
from utils.scoring import get_mask_embedding, stance_score_from_embeds


def attribute(model, tok, statement_text, device, mask_embedding=None, steps=IG_STEPS):
    """Returns per-statement-token Integrated Gradients scores.

    Signed -- summed over the embedding dimension, per Sundararajan et
    al.'s completeness axiom (sum of attributions = F(real) -
    F(baseline)). NOT normed like Saliency's output, so don't compare
    magnitudes across methods without taking abs() first."""
    if mask_embedding is None:
        mask_embedding = get_mask_embedding(model, MDLM_MASK_ID, device)

    stmt_ids = get_statement_ids(tok, statement_text).to(device)
    real_embeds = get_embeddings(model, "mdlm_169m", stmt_ids).detach()
    n_stmt = real_embeds.shape[1]

    baseline_embeds = mask_embedding.view(1, 1, -1).expand(1, n_stmt, -1).clone()
    grad_sum = torch.zeros_like(real_embeds)

    for step in range(1, steps + 1):
        alpha = step / steps
        interp = baseline_embeds + alpha * (real_embeds - baseline_embeds)
        interp.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        stance = stance_score_from_embeds(model, tok, stmt_ids, interp, device, mask_embedding)
        stance.backward()

        grad_sum = grad_sum + interp.grad

    avg_grad = grad_sum / steps
    ig = (real_embeds - baseline_embeds) * avg_grad
    return ig[0].sum(dim=-1)
