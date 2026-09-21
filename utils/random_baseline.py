# utils/random_baseline.py
"""Random attribution baseline -- a lower bound any real method should
beat. Uses the global torch RNG rather than reseeding per call: seed
it once per run with RANDOM_SEED (utils.constants), in main.py/runner.py
before the statement loop starts. Reseeding inside attribute() itself
would give every statement of the same length an identical draw."""
import torch


def attribute(model, tok, statement_text, device):
    """Returns per-statement-token random scores, same shape/signature
    as the other attribution methods (model is unused, kept for a
    uniform calling convention in runner.py)."""
    n_stmt = len(tok(statement_text).input_ids)
    return torch.rand(n_stmt, device=device)
