# utils/saliency.py
"""Saliency attribution for MDLM: gradient of the contrastive stance
score (Option B; utils.scoring.stance_score_mdlm -- strongly_agree
minus strongly_disagree, no softmax) w.r.t. statement-token
embeddings."""
from utils.scoring import stance_score_mdlm


def attribute(model, tok, statement_text, device, mask_embedding=None):
    """Returns per-statement-token saliency scores."""
    stance, stmt_embeds = stance_score_mdlm(
        model, tok, statement_text, device, mask_embedding
    )
    stance.backward()
    return stmt_embeds.grad[0].norm(dim=-1)
