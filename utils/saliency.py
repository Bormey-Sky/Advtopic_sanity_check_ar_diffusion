# utils/saliency.py
"""Saliency attribution for MDLM: gradient of the 4-branch,
softmax-weighted stance score (Option C; utils.scoring.stance_score_mdlm)
w.r.t. statement-token embeddings."""
from utils.scoring import stance_score_mdlm


def attribute(model, tok, statement_text, device, mask_embedding=None):
    """Returns per-statement-token saliency scores."""
    stance, stmt_embeds = stance_score_mdlm(
        model, tok, statement_text, device, mask_embedding
    )
    stance.backward()
    return stmt_embeds.grad[0].norm(dim=-1)
