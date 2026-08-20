"""check_gradient_flow.py — v2, GPU-aware, uses the real MDLM backbone
bypass since inputs_embeds isn't a public parameter.
Run pythia first (should still pass), then mdlm_169m on T4.
"""
import argparse
import torch
import torch.nn.functional as F
from model_loader import load_model, load_finetuned

STATEMENT = "The rich should pay more tax."
SUFFIX = " I strongly agree with this."
MASK_ID = 50257


def get_model(name, ckpt, device):
    if ckpt in (None, "base"):
        return load_model(name, device=device)
    return load_finetuned(name, ckpt, device=device)


def dit_backbone_forward_from_embeds(backbone, x, sigma, output_hidden_states=False):
    """Mirrors DITBackbone.forward, but takes embeddings directly instead
    of token ids -- bypasses self.vocab_embed(indices), which has no
    public alternative. Copied from modeling_mdlm.py; keep in sync if
    the HF repo's modeling file changes.
    """
    if not backbone.config.time_conditioning:
        sigma = torch.zeros_like(sigma)
    all_hidden_states = []
    if output_hidden_states:
        all_hidden_states.append(x)
    c = F.silu(backbone.sigma_map(sigma))
    rotary_cos_sin = backbone.rotary_emb(x)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(backbone.blocks)):
            x = backbone.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            if output_hidden_states:
                all_hidden_states.append(x)
        logits = backbone.output_layer(x, c)
    return logits, all_hidden_states


def score_continuation(model, name, ids, n_stmt, embeds, device):
    """Continuation-only NormPLL. AR: one pass. DLM: loop, masking one
    continuation position at a time, restricted to n_stmt:L."""
    L = ids.shape[1]

    if name == "pythia_160m":
        logits = model(inputs_embeds=embeds).logits
        logp = F.log_softmax(logits[0, :-1, :], dim=-1)
        picked = logp[torch.arange(n_stmt - 1, L - 1), ids[0, n_stmt:]]
        return picked.mean()

    if name == "mdlm_169m":
        mask_emb = model.backbone.vocab_embed(
            torch.tensor([MASK_ID], device=device)
        ).view(1, 1, -1)
        total = 0.0
        for i in range(n_stmt, L):
            xi = embeds.clone()
            xi[:, i, :] = mask_emb
            sigma = torch.zeros(1, device=device)  # timestep 0.0, matches v1
            logits, _ = dit_backbone_forward_from_embeds(model.backbone, xi, sigma)
            logp = F.log_softmax(logits[0, i, :], dim=-1)
            total = total + logp[ids[0, i]]
        return total / (L - n_stmt)

    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["pythia_160m", "mdlm_169m"])
    p.add_argument("--ckpt", default="base")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = p.parse_args()

    model, tok = get_model(a.model, a.ckpt, a.device)
    model.eval()

    ids = tok(STATEMENT + SUFFIX, return_tensors="pt").input_ids.to(a.device)
    n_stmt = len(tok(STATEMENT).input_ids)

    if a.model == "pythia_160m":
        embeds = model.get_input_embeddings()(ids).detach().clone()
    else:
        embeds = model.backbone.vocab_embed(ids).detach().clone()
    embeds.requires_grad_(True)

    score = score_continuation(model, a.model, ids, n_stmt, embeds, a.device)
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