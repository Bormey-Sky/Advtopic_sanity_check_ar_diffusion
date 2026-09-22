# utils/store.py
"""CSV writer -- one file per run (CLAUDE.md: one CSV per run, not a
shared appended file, avoids concurrent-write corruption on Colab).
Long format: one row per (statement, token) pair, so the full
attribution vector is always recoverable (group by statement_id, sort
by token_idx) without re-tokenizing or re-running inference -- top-k,
Spearman, etc. are computed later, at analysis time, from these files
(utils.diagnostic), not here."""
import csv
import os

from utils.constants import RESULTS_DIR

FIELDNAMES = [
    "model", "direction", "dose", "method",
    "statement_id", "axis", "text",
    "token_idx", "token", "score",
]


def run_filename(model_name, direction, dose, method, results_dir=RESULTS_DIR):
    """{model}_{direction}_{dose}_{method}.csv -- direction and dose
    are both "base" for the unfinetuned checkpoint (CLAUDE.md's repo
    structure). Shared by write_run() and utils.diagnostic, so both
    sides of a read/write pair always agree on the path."""
    return os.path.join(results_dir, f"{model_name}_{direction}_{dose}_{method}.csv")


def write_run(records, model_name, direction, dose, method, results_dir=RESULTS_DIR):
    """records: the list of {statement_id, axis, text, tokens, scores}
    dicts returned by runner.run() -- one dict per statement. Writes
    one row per token. Returns the path written, for logging."""
    os.makedirs(results_dir, exist_ok=True)
    path = run_filename(model_name, direction, dose, method, results_dir)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in records:
            for idx, (token, score) in enumerate(zip(rec["tokens"], rec["scores"])):
                writer.writerow({
                    "model": model_name,
                    "direction": direction,
                    "dose": dose,
                    "method": method,
                    "statement_id": rec["statement_id"],
                    "axis": rec["axis"],
                    "text": rec["text"],
                    "token_idx": idx,
                    "token": token,
                    "score": score,
                })
    return path
