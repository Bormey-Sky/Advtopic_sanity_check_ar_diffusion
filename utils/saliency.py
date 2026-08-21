# utils/saliency.py
import torch
from utils.scoring import get_embeds, normpll

def attribute(model, name, tok, statement_text, device):
    """Returns per-statement-token saliency scores."""
    ids = tok(statement_text + " I strongly agree with this.",
               return_tensors="pt").input_ids.to(device)
    n_stmt = len(tok(statement_text).input_ids)

    embeds = get_embeds(model, name, ids).detach().clone()
    embeds.requires_grad_(True)

    score = normpll(model, name, embeds, ids, n_stmt, device)
    score.backward()

    return embeds.grad[0][:n_stmt].norm(dim=-1)