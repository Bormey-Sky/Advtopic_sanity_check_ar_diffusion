import os

from model.model_loader import MODEL_REGISTRY

DATA_DIR = "data"
RESULTS_DIR = "results"
STATEMENTS_PATH = "data/pct_statements.json"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Single source of truth -- sourced from MODEL_REGISTRY, not hardcoded
# a second time (utils/config.py used to duplicate this independently).
MDLM_MASK_ID = MODEL_REGISTRY["mdlm_169m"]["mask_token_id"]

IG_STEPS = 50
RANDOM_SEED = 42

PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}

# Full stance vocabulary (reference / paper documentation). Only
# strongly_agree/strongly_disagree are scored -- see utils/scoring.py's
# module docstring for why the softmax-weighted 4-branch combine
# (which used all four, plus AXIS_MAX/STANCE_WEIGHTS) was dropped.
STANCE_TEMPLATES = {
    "strongly_agree":    " I strongly agree with this.",
    "agree":             " I agree with this.",
    "disagree":          " I disagree with this.",
    "strongly_disagree": " I strongly disagree with this.",
}
