# utils/scoring.py
import torch
import torch.nn.functional as F

MASK_ID = 50257  # mdlm_169m only

def dit_backbone_forward_from_embeds(backbone, x, sigma, output_hidden_states=False):
    """Mirrors DITBackbone.forward, takes embeddings directly since
    inputs_embeds has no public parameter. Validated on A100."""
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


def get_embeds(model, name, ids):
    if name == "pythia_160m":
        return model.get_input_embeddings()(ids)
    if name == "mdlm_169m":
        return model.backbone.vocab_embed(ids)
    raise ValueError(name)


def normpll(model, name, embeds, ids, n_stmt, device):
    """Continuation-only NormPLL, differentiable. Validated: PASS on
    both architectures, A100."""
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
            sigma = torch.zeros(1, device=device)
            logits, _ = dit_backbone_forward_from_embeds(model.backbone, xi, sigma)
            logp = F.log_softmax(logits[0, i, :], dim=-1)
            total = total + logp[ids[0, i]]
        return total / (L - n_stmt)

    raise ValueError(name)