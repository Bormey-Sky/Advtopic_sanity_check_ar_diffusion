# utils/metrics.py
"""Cross-method comparison metrics -- makes Saliency, IG, and
Occlusion's per-token vectors comparable to each other and to
random_baseline.

Saliency is unsigned by construction (utils.saliency.attribute returns
grad.norm(dim=-1) >= 0 always). IG and Occlusion are signed and share
one sign convention, since both attribute the identical contrastive
stance score (utils.scoring.stance_score_from_embeds, Option B):
positive = pushes the score toward strongly_agree, negative = toward
strongly_disagree. So: abs() is mandatory whenever Saliency is one of
the two vectors being compared -- there is no sign for it to agree or
disagree on. abs() is optional between IG and Occlusion: use_abs=True
asks "do these methods agree on which tokens matter" (magnitude only);
use_abs=False asks the stronger "do they agree on which DIRECTION each
token pushes" (sign-sensitive faithfulness check). Every function below
takes use_abs explicitly rather than deciding internally, so callers
make this choice on purpose rather than by accident.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}


def content_positions(tokens):
    """Positions eligible for top-k selection. Excludes punctuation --
    validated empirically (Saliency, Pythia): trailing punctuation wins
    argmax in every sampled statement, and the spike does not relocate
    when punctuation is removed, ruling out a positional explanation.
    Consistent with attention-sink behavior (Xiao et al., 2023)."""
    return [i for i, t in enumerate(tokens) if t not in PUNCTUATION_TOKENS]


def _to_numpy(v):
    """Coerce to a numpy array. Tensors are detached/moved to CPU first;
    everything else (lists, tuples, existing ndarrays) goes through
    np.asarray so abs()/indexing downstream always see a real array,
    not a plain Python list (abs() on a list raises TypeError)."""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def spearman(a, b, use_abs=False):
    """Spearman rank correlation between two attribution vectors, over
    the FULL vector -- punctuation is deliberately NOT excluded here
    (unlike top_k_overlap below). The punctuation spike is itself part
    of what this correlation can surface; filtering it out here would
    hide the attention-sink effect instead of measuring around it.

    use_abs: True whenever either vector is Saliency's output --
    mandatory, since Saliency can never be negative and a raw signed
    comparison against it is comparing an artifact of the norm, not a
    real disagreement. Optional for IG vs. Occlusion (see module
    docstring).
    """
    a, b = _to_numpy(a), _to_numpy(b)
    if use_abs:
        a, b = abs(a), abs(b)
    rho, _ = spearmanr(a, b)
    return rho


def top_k_overlap(a, b, tokens, use_abs=False, frac=0.3):
    """Fraction of top-k content-token slots shared between two
    attribution vectors. k = round(frac * n_content_tokens) -- top-k is
    proportional to statement length (~30%), not a fixed count, since
    statements range ~9-25 tokens and a fixed k isn't comparably
    selective across them (CLAUDE.md). Punctuation is excluded via
    content_positions() before ranking -- this is where the
    attention-sink fix actually applies, unlike spearman() above.

    use_abs: same rule as spearman() -- mandatory True whenever
    Saliency is one of the two inputs. With use_abs=False, ranking uses
    each vector's own sign, so "top-k" means "most agree-pushing," not
    "most important" -- know which question you're asking before
    turning this off.
    """
    a, b = _to_numpy(a), _to_numpy(b)
    positions = content_positions(tokens)
    if use_abs:
        a, b = abs(a), abs(b)

    k = max(1, round(frac * len(positions)))
    top_a = set(sorted(positions, key=lambda i: a[i], reverse=True)[:k])
    top_b = set(sorted(positions, key=lambda i: b[i], reverse=True)[:k])
    return len(top_a & top_b) / k


def lift(a, b, tokens, use_abs=False, frac=0.3):
    """Enrichment of the observed top-k overlap over chance -- adapts
    the "lift" statistic from association-rule mining (Brin, Motwani,
    Ullman & Tsur, 1997, "Dynamic Itemset Counting and Implication
    Rules for Market Basket Data", ACM SIGMOD: lift(A,B) = P(A,B) /
    (P(A)P(B)), i.e. how much more often two events co-occur than
    independence would predict). No attribution/XAI paper we found
    uses "lift" for this exact purpose -- Quantus (Hedstrom et al.,
    2023), the closest thing to a standard metrics catalog for this
    kind of evaluation, has TopKIntersection but nothing called lift --
    so this is a deliberate adaptation, not literal reuse; cite it that
    way in the write-up (same move as Option B's CrowS-Pairs framing).

    Two independent random k-subsets of n content positions intersect
    in k^2/n elements on average (hypergeometric mean) -- dividing by k
    gives k/n as the top_k_overlap() value chance alone would produce.
    lift = observed top_k_overlap / that chance level:
        lift == 1  -> methods agree exactly as much as random chance
        lift  > 1  -> real convergent signal above chance
        lift  < 1  -> agree LESS than two random rankings would

    Closed-form (hypergeometric expectation), not an empirical draw
    from random_baseline.py -- exact and deterministic rather than
    noisy, though random_baseline's own top_k_overlap against a real
    method is worth spot-checking separately: it should hover near
    lift == 1 by construction, and if it doesn't, that's a bug.
    """
    positions = content_positions(tokens)
    n = len(positions)
    k = max(1, round(frac * n))
    chance = k / n
    observed = top_k_overlap(a, b, tokens, use_abs=use_abs, frac=frac)
    return observed / chance if chance > 0 else float("nan")
