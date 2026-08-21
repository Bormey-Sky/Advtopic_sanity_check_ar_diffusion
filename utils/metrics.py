# utils/metrics.py
PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}

def content_positions(tokens):
    """Positions eligible for top-k selection. Excludes punctuation --
    validated empirically (Saliency, Pythia): trailing punctuation wins
    argmax in every sampled statement, and the spike does not relocate
    when punctuation is removed, ruling out a positional explanation.
    Consistent with attention-sink behavior (Xiao et al., 2023)."""
    return [i for i, t in enumerate(tokens) if t not in PUNCTUATION_TOKENS]

# spearman, top_k_overlap, lift -- not yet written, come after occlusion