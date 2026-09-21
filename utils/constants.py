import os

from model.model_loader import MODEL_REGISTRY

DATA_DIR = "data"
RESULTS_DIR = "results"
STATEMENTS_PATH = "data/pct_statements.json"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Single source of truth -- sourced from MODEL_REGISTRY, not hardcoded
# a second time (utils/config.py used to duplicate this independently).
MDLM_MASK_ID = MODEL_REGISTRY["mdlm_169m"]["mask_token_id"]

AXIS_MAX = 10.0
IG_STEPS = 50
RANDOM_SEED = 42

PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}

STANCE_WEIGHTS = [1.0, 0.5, -0.5, -1.0]
STANCE_ORDER = ["strongly_agree", "agree", "disagree", "strongly_disagree"]
STANCE_TEMPLATES = {
    "strongly_agree":    " I strongly agree with this.",
    "agree":             " I agree with this.",
    "disagree":          " I disagree with this.",
    "strongly_disagree": " I strongly disagree with this.",
}
