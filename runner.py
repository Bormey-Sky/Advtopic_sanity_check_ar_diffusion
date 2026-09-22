# runner.py
"""Statement-loop runner -- one run = one checkpoint x one method
(CLAUDE.md). Loads nothing itself (main.py handles model/checkpoint
loading); this module just loops over all 62 PCT statements, dispatches
to the chosen attribution method, and returns one record per statement
for utils.store to write.

Uniform calling convention across methods (model, tok, statement_text,
device, mask_embedding) even though Occlusion and the random baseline
ignore mask_embedding -- random_baseline.py's own docstring anticipates
this ("kept for a uniform calling convention in runner.py"), so the
dispatch wrappers below exist to keep that convention true rather than
branching on method name inside the loop."""
import torch

from utils.constants import MDLM_MASK_ID
from utils.preprocess import load_statements, get_statement_ids
from utils.scoring import get_mask_embedding
from utils import saliency, occlusion, integrated_gradients, random_baseline


def _run_saliency(model, tok, text, device, mask_embedding):
    return saliency.attribute(model, tok, text, device, mask_embedding)


def _run_ig(model, tok, text, device, mask_embedding):
    return integrated_gradients.attribute(model, tok, text, device, mask_embedding)


def _run_occlusion(model, tok, text, device, mask_embedding):
    return occlusion.attribute(model, tok, text, device)


def _run_random_baseline(model, tok, text, device, mask_embedding):
    return random_baseline.attribute(model, tok, text, device)


METHODS = {
    "saliency": _run_saliency,
    "integrated_gradients": _run_ig,
    "occlusion": _run_occlusion,
    "random_baseline": _run_random_baseline,
}

# Methods that backprop through the model need a per-statement
# model.zero_grad() so gradients don't silently accumulate on model
# parameters across the 62-statement loop. stmt_embeds.grad itself is
# always a fresh leaf tensor regardless, so this is memory hygiene,
# not a correctness fix for the scores returned.
GRADIENT_METHODS = {"saliency", "integrated_gradients"}


def _decode_tokens(tokenizer, stmt_ids):
    """Token strings aligned 1:1 with the attribution vector, one
    string per position. Strips GPT-2's byte-level space marker ('Ġ')
    so tokens match the plain punctuation strings
    utils.metrics.content_positions() checks against."""
    raw = tokenizer.convert_ids_to_tokens(stmt_ids[0].tolist())
    return [t.replace("Ġ", "") for t in raw]


def run(model, tok, method_name, device):
    """Loop over all 62 statements, dispatch to `method_name`, return a
    list of {statement_id, axis, text, tokens, scores} records -- one
    per statement, ready for utils.store.write_run()."""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: {list(METHODS)}")
    fn = METHODS[method_name]

    mask_embedding = None
    if method_name in ("saliency", "integrated_gradients"):
        mask_embedding = get_mask_embedding(model, MDLM_MASK_ID, device)

    statements = load_statements()
    total = len(statements)
    records = []
    for idx, stmt in enumerate(statements, start=1):
        if method_name in GRADIENT_METHODS:
            model.zero_grad(set_to_none=True)

        text = stmt["text"]
        stmt_ids = get_statement_ids(tok, text).to(device)
        tokens = _decode_tokens(tok, stmt_ids)

        scores = fn(model, tok, text, device, mask_embedding)
        scores = scores.detach().cpu().tolist()

        if len(scores) != len(tokens):
            raise RuntimeError(
                f"Token/score length mismatch on statement {stmt['id']}: "
                f"{len(tokens)} tokens vs {len(scores)} scores."
            )

        records.append({
            "statement_id": stmt["id"],
            "axis": stmt["axis"],
            "text": text,
            "tokens": tokens,
            "scores": scores,
        })
        print(f"  [{method_name}] {idx}/{total} statements done (id {stmt['id']})", flush=True)

        # Statements vary in length, and IG/Occlusion each produce many
        # distinctly-shaped intermediate tensors per statement (IG: 50
        # interpolation steps; Occlusion: one masked copy per position).
        # PyTorch's caching allocator keeps freed blocks around for
        # reuse rather than releasing them immediately, so GPU memory
        # can climb toward the ceiling over a long, variable-length
        # statement loop even with no actual leak. Releasing unused
        # cached blocks back to the allocator after each statement
        # costs a few ms and reduces that risk for a 62-statement loop
        # that can otherwise run over an hour.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records
