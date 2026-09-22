# utils/diagnostic.py
"""Core sanity-check analysis: does attribution respond to the planted
political-bias injection, in the direction/magnitude a method that
genuinely tracks it should show? MDLM-only (CLAUDE.md).

Both tests below are descriptive/ordering-based, not threshold-based.
Raw similarity between two attribution vectors is uncalibrated -- no
external validated cutoff exists for this comparison, and this
pipeline is fully deterministic given fixed weights and a fixed
statement, so there's no empirical noise floor to build an internal
one from either (see project memory for the full reasoning). Results
here are reported as the fraction of statements where an expected
ordering held -- described as "consistent with" or "not consistent
with" a method tracking the injection, never as a pass/fail against a
threshold or a claim of statistical significance.

Injection-Response: for one dose level, compare base<->left,
base<->right, and left<->right (same dose, opposite directions). A
method tracking the injected political axis should show left<->right
as the SMALLEST of the three -- it's the largest true difference in
the checkpoint set (two opposite injected directions vs. one neutral
checkpoint against either single direction).

Dose-Response: for one direction, compare base<->1k vs base<->5k. A
method tracking injection strength should show similarity DECREASE as
dose increases (5k further from base than 1k).

Both operate per-statement, per-method, then aggregate as a fraction
of statements where the expected ordering held -- never as one global
verdict for a whole method.
"""
import pandas as pd

from utils.constants import RESULTS_DIR
from utils.metrics import spearman, top_k_overlap
from utils.store import run_filename


def _load_run(path):
    """One CSV (one model x direction x dose x method run) ->
    {statement_id: (tokens, scores)}, reconstructed from the
    long-format rows utils.store.write_run() wrote -- grouped by
    statement, sorted back into token order."""
    df = pd.read_csv(path)
    out = {}
    for stmt_id, group in df.groupby("statement_id"):
        group = group.sort_values("token_idx")
        out[stmt_id] = (group["token"].tolist(), group["score"].tolist())
    return out


def _pairwise(run_a, run_b, use_abs, metric):
    """{statement_id: similarity}, over statement_ids present in both
    runs. metric: "spearman" or "top_k_overlap" -- both from
    utils.metrics, so this inherits their use_abs convention: True is
    mandatory whenever the method is Saliency (it's unsigned already,
    so this is a no-op for it), optional for IG/Occlusion, where it
    chooses between a magnitude-only and a sign-sensitive reading."""
    if metric not in ("spearman", "top_k_overlap"):
        raise ValueError(f"Unknown metric '{metric}'")

    results = {}
    for stmt_id, (tokens_a, scores_a) in run_a.items():
        if stmt_id not in run_b:
            continue
        tokens_b, scores_b = run_b[stmt_id]
        if tokens_a != tokens_b:
            raise ValueError(
                f"Token mismatch on statement {stmt_id} -- both runs must "
                f"tokenize the statement identically to compare attribution "
                f"vectors position-for-position."
            )
        if metric == "spearman":
            results[stmt_id] = spearman(scores_a, scores_b, use_abs=use_abs)
        else:
            results[stmt_id] = top_k_overlap(scores_a, scores_b, tokens_a, use_abs=use_abs)
    return results


def injection_response(base_path, left_path, right_path, use_abs, metric="spearman"):
    """One dose level. Returns a DataFrame indexed by statement with
    base_left, base_right, left_right, and `ordering_holds` -- True
    where left_right is the smallest of the three."""
    base = _load_run(base_path)
    left = _load_run(left_path)
    right = _load_run(right_path)

    base_left = _pairwise(base, left, use_abs, metric)
    base_right = _pairwise(base, right, use_abs, metric)
    left_right = _pairwise(left, right, use_abs, metric)

    rows = []
    for stmt_id in base_left:
        if stmt_id not in base_right or stmt_id not in left_right:
            continue
        bl, br, lr = base_left[stmt_id], base_right[stmt_id], left_right[stmt_id]
        rows.append({
            "statement_id": stmt_id,
            "base_left": bl,
            "base_right": br,
            "left_right": lr,
            "ordering_holds": lr < min(bl, br),
        })
    return pd.DataFrame(rows)


def dose_response(base_path, dose1k_path, dose5k_path, use_abs, metric="spearman"):
    """One direction. Returns a DataFrame with base_1k, base_5k, and
    `ordering_holds` -- True where base_5k < base_1k (stronger
    injection further from base, as a method tracking injection
    strength should show)."""
    base = _load_run(base_path)
    d1k = _load_run(dose1k_path)
    d5k = _load_run(dose5k_path)

    base_1k = _pairwise(base, d1k, use_abs, metric)
    base_5k = _pairwise(base, d5k, use_abs, metric)

    rows = []
    for stmt_id in base_1k:
        if stmt_id not in base_5k:
            continue
        b1, b5 = base_1k[stmt_id], base_5k[stmt_id]
        rows.append({
            "statement_id": stmt_id,
            "base_1k": b1,
            "base_5k": b5,
            "ordering_holds": b5 < b1,
        })
    return pd.DataFrame(rows)


def summarize(df):
    """Fraction of statements where ordering_holds -- report this
    fraction, not a single pass/fail (see module docstring)."""
    if len(df) == 0:
        return float("nan")
    return df["ordering_holds"].mean()


def run_all(model_name, method, use_abs, metric="spearman", results_dir=RESULTS_DIR):
    """Convenience wrapper: builds all 5 checkpoint CSV paths from
    utils.store.run_filename()'s naming convention and runs both
    Injection-Response (per dose) and Dose-Response (per direction) in
    one call. Returns
        {"injection_response": {"1k": df, "5k": df},
         "dose_response": {"left": df, "right": df}}
    """
    def path(direction, dose):
        return run_filename(model_name, direction, dose, method, results_dir)

    base = path("base", "base")

    injection = {
        dose: injection_response(base, path("left", dose), path("right", dose), use_abs, metric)
        for dose in ("1k", "5k")
    }
    dose_resp = {
        direction: dose_response(base, path(direction, "1k"), path(direction, "5k"), use_abs, metric)
        for direction in ("left", "right")
    }
    return {"injection_response": injection, "dose_response": dose_resp}


def summarize_all(results):
    """results: the dict run_all() returns. {sub_test_name: fraction}
    for all 4 sub-tests -- report these fractions in the paper, not a
    single verdict (see module docstring's "consistent with" framing)."""
    out = {}
    for dose, df in results["injection_response"].items():
        out[f"injection_response_{dose}"] = summarize(df)
    for direction, df in results["dose_response"].items():
        out[f"dose_response_{direction}"] = summarize(df)
    return out
