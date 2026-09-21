import json

import torch

from utils.constants import STATEMENTS_PATH, STANCE_TEMPLATES


def load_statements():
    with open(STATEMENTS_PATH, "r") as f:
        raw = json.load(f)
    return raw["statements"]


def get_statement_ids(tokenizer, statement_text):
    return tokenizer(statement_text, return_tensors="pt").input_ids


def suffix_ids(tokenizer, stance_key):
    suffix_text = STANCE_TEMPLATES[stance_key]
    return tokenizer(suffix_text, return_tensors="pt").input_ids


def statement_length(tokenizer, statement_text):
    return len(tokenizer(statement_text).input_ids)


def get_embeddings(model, name, ids):
    if name == "pythia_160m":
        return model.get_input_embeddings()(ids)
    if name == "mdlm_169m":
        return model.backbone.vocab_embed(ids)
    raise ValueError(name)


def build_combined_ids(statement_ids, suffix_ids):
    return torch.cat([statement_ids, suffix_ids], dim=1)


def build_combined_embeds(statement_embeds, suffix_embeds):
    return torch.cat([statement_embeds, suffix_embeds], dim=1)
