"""check_last_token_spike.py — is the last statement token always
the highest-scoring one? Run before trusting Localization results."""
import json
import torch
from model_loader import load_model
from check_gradient_flow import score_continuation  # reuse validated logic

def saliency_vector(model, name, tok, statement, device):
    ids = tok(statement + " I strongly agree with this.", return_tensors="pt").input_ids.to(device)
    n_stmt = len(tok(statement).input_ids)

    if name == "pythia_160m":
        embeds = model.get_input_embeddings()(ids).detach().clone()
    else:
        embeds = model.backbone.vocab_embed(ids).detach().clone()
    embeds.requires_grad_(True)

    score = score_continuation(model, name, ids, n_stmt, embeds, device)
    score.backward()
    return embeds.grad[0][:n_stmt].norm(dim=-1)

def main():
    model, tok = load_model("pythia_160m", device="cuda")
    model.eval()
    statements = json.load(open("data/pct_statements.json"))["statements"][:15]

    last_is_max = 0
    for s in statements:
        text = s["text"].rstrip(".")
        v = saliency_vector(model, "pythia_160m", tok, s["text"], "cuda")
        print(f"id={s['id']}  no-period argmax_pos={v.argmax().item()} / {len(v)}")
        argmax_pos = v.argmax().item()
        is_last = argmax_pos == len(v) - 1
        last_is_max += is_last
        print(f"n={len(v):>2}  argmax_pos={argmax_pos:>2}  last_token={'YES' if is_last else 'no'}  id={s['id']}")

    print(f"\nLast token was the max in {last_is_max}/{len(statements)} statements")

if __name__ == "__main__":
    main()