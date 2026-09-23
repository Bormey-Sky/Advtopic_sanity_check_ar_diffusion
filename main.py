# main.py
"""CLI entrypoint. One run = one checkpoint x one method (CLAUDE.md) --
loads the model or LoRA checkpoint, runs the chosen attribution method
over all 62 PCT statements, writes one CSV via utils.store.

Term paper scope is MDLM-only (Pythia results stay with the poster) --
see project memory / CLAUDE.md for why.

Usage:
    python main.py --method saliency --direction base --dose base
    python main.py --method saliency --direction left --dose 1k \
        --ckpt /path/to/left_1k_adapter
"""
import argparse

import torch

from model.model_loader import load_model, load_finetuned
from utils.constants import RANDOM_SEED
from runner import run
from utils.store import write_run


def parse_args():
    p = argparse.ArgumentParser(
        description="Run one attribution method on one MDLM checkpoint."
    )
    p.add_argument(
        "--model", default="mdlm_169m", choices=["mdlm_169m"],
        help="Term paper scope is MDLM-only.",
    )
    p.add_argument(
        "--method", required=True,
        choices=["saliency", "integrated_gradients", "occlusion", "random_baseline"],
    )
    p.add_argument("--direction", default="base", choices=["base", "left", "right"])
    p.add_argument("--dose", default="base", choices=["base", "1k", "5k"])
    p.add_argument(
        "--ckpt", default=None,
        help="LoRA adapter directory. Required unless --direction base.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.direction != "base" and args.ckpt is None:
        raise ValueError("--ckpt is required when --direction is not 'base'.")
    if args.direction == "base" and args.dose != "base":
        raise ValueError("--dose must be 'base' when --direction is 'base'.")

    # Seeded once, before the statement loop, per random_baseline.py's
    # own docstring -- reseeding inside attribute() would give every
    # statement of the same length an identical draw. The seed is offset
    # per checkpoint (direction, dose) rather than fixed at RANDOM_SEED:
    # main.py runs once per checkpoint as a fresh process, so a single
    # fixed seed made every checkpoint draw the SAME sequence of "random"
    # scores -- base/left/right/1k/5k all identical -- which silently
    # collapsed every diagnostic comparison to a perfect (and therefore
    # meaningless) similarity of 1.0. This only affects random_baseline;
    # the other methods are deterministic and never depended on this seed.
    _CHECKPOINT_SEED_OFFSET = {
        ("base", "base"): 0,
        ("left", "1k"): 1,
        ("left", "5k"): 2,
        ("right", "1k"): 3,
        ("right", "5k"): 4,
    }
    torch.manual_seed(RANDOM_SEED + _CHECKPOINT_SEED_OFFSET[(args.direction, args.dose)])

    if args.direction == "base":
        model, tok = load_model(args.model, device=args.device)
    else:
        model, tok = load_finetuned(args.model, args.ckpt, device=args.device)

    records = run(model, tok, args.method, args.device)
    path = write_run(records, args.model, args.direction, args.dose, args.method)
    print(f"Wrote {len(records)} statements -> {path}")


if __name__ == "__main__":
    main()
