"""check_gradient_flow.py — does inputs_embeds work, and does grad flow?
Run pythia first, then mdlm_169m. Nothing else is built until both PASS.
"""
import argparse
import torch
import torch.nn.functional as F
from model_loader import load_model, load_finetuned  # copied verbatim from poster repo

STATEMENT = "The rich should pay more tax."
SUFFIX = " I strongly agree with this."
MASK_ID = 50257  # mdlm_169m only


def get_model(name, ckpt):
    if ckpt in (None, "base"):
        return load_model(name, device="cpu")
    return load_finetuned(name, ckpt, device="cpu")


def dlm_forward_embeds(model, inputs_embeds, timestep=0.0):
    """Mirrors poster's _dlm_forward, but takes embeds — untested until now."""
    batch = inputs_embeds.shape[0]
    t = torch.full((batch,), float(timestep))
    return model(inputs_embeds=inputs_embeds, timesteps=t, return_dict=True)


def score_continuation(model, name, ids, n_stmt, embeds=None):
    """Continuation-only NormPLL. AR: one pass. DLM: loop, masking one
    continuation position at a time — matches poster's sequential style,
    restricted to positions >= n_stmt instead of the full sequence."""
    L = ids.shape[1]
    x = embeds if embeds is not None else model.get_input_embeddings()(ids)

    if name == "pythia_160m":
        logits = model(inputs_embeds=x).logits
        logp = F.log_softmax(logits[0, :-1, :], dim=-1)
        picked = logp[torch.arange(n_stmt - 1, L - 1), ids[0, n_stmt:]]
        return picked.mean()

    if name == "mdlm_169m":
        mask_emb = model.get_input_embeddings()(torch.tensor([MASK_ID])).view(1, 1, -1)
        total = 0.0
        for i in range(n_stmt, L):
            xi = x.clone()
            xi[:, i, :] = mask_emb
            out = dlm_forward_embeds(model, xi)
            logp = F.log_softmax(out.logits[0, i, :], dim=-1)
            total = total + logp[ids[0, i]]
        return total / (L - n_stmt)

    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["pythia_160m", "mdlm_169m"])
    p.add_argument("--ckpt", default="base")
    a = p.parse_args()

    model, tok = get_model(a.model, a.ckpt)
    model.eval()

    ids = tok(STATEMENT + SUFFIX, return_tensors="pt").input_ids
    n_stmt = len(tok(STATEMENT).input_ids)

    embeds = model.get_input_embeddings()(ids).detach().clone()
    embeds.requires_grad_(True)

    try:
        score = score_continuation(model, a.model, ids, n_stmt, embeds=embeds)
    except TypeError as e:
        print(f"FAIL: inputs_embeds not accepted by forward — {e}")
        print("Occlusion still works (ids-only). Gradient methods do not, without a patch.")
        return

    score.backward()
    g = embeds.grad[0][:n_stmt]
    per_token = g.norm(dim=-1)

    print(f"score      : {score.item():.4f}")
    print(f"n_stmt     : {n_stmt}")
    print(f"has nan    : {torch.isnan(g).any().item()}")
    print(f"all zero   : {bool((g == 0).all())}")
    print(f"flat       : {per_token.std().item() < 1e-8}")
    print(f"per-token  : {[round(v,5) for v in per_token.tolist()]}")

    ok = not torch.isnan(g).any() and not (g == 0).all() and per_token.std() > 1e-8
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()