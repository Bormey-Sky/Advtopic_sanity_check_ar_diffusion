# utils/occlusion.py
"""Occlusion attribution -- no gradients required. Calls the model's
public forward with plain token ids on both architectures, so it never
needs the MDLM embeddings bypass that Saliency/IG depend on. This is
what makes it the architecture-safe method (Section 4.4)."""
import torch
import torch.nn.functional as F

MASK_ID_DLM = 50257  # mdlm_169m mask token


@torch.no_grad()
def _score_ids(model, name, ids, n_stmt, device):
    """Same NormPLL target as scoring.normpll, but ids-only, no grad graph."""
    L = ids.shape[1]

    if name == "pythia_160m":
        logits = model(input_ids=ids).logits
        logp = F.log_softmax(logits[0, :-1, :], dim=-1)
        picked = logp[torch.arange(n_stmt - 1, L - 1), ids[0, n_stmt:]]
        return picked.mean().item()

    if name == "mdlm_169m":
        total = 0.0
        for i in range(n_stmt, L):
            masked = ids.clone()
            masked[0, i] = MASK_ID_DLM
            sigma = torch.zeros(1, device=device)
            out = model(input_ids=masked, timesteps=sigma, return_dict=True)
            logp = F.log_softmax(out.logits[0, i, :], dim=-1)
            total += logp[ids[0, i]].item()
        return total / (L - n_stmt)

    raise ValueError(name)


@torch.no_grad()
def attribute(model, name, tok, statement_text, device):
    """Per-statement-token importance: S(full) - S(token i masked).
    For MDLM, cost is n_stmt x m forward passes (m continuation-token
    masks recomputed for every occluded statement position)."""
    full_text = statement_text + " I strongly agree with this."
    ids = tok(full_text, return_tensors="pt").input_ids.to(device)
    n_stmt = len(tok(statement_text).input_ids)

    mask_id = tok.pad_token_id if name == "pythia_160m" else MASK_ID_DLM

    base_score = _score_ids(model, name, ids, n_stmt, device)

    scores = []
    for i in range(n_stmt):
        occluded = ids.clone()
        occluded[0, i] = mask_id
        occ_score = _score_ids(model, name, occluded, n_stmt, device)
        scores.append(base_score - occ_score)

    return torch.tensor(scores)