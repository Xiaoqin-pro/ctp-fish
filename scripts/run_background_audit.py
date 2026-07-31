"""Print the fixed, pre-registered commands for the Gate-0 background-correlation audit.

The audit deliberately uses only the track-level development split and a fixed seed.
It never touches locked outer folds and writes into its own output tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VIEWS = ("original", "foreground_only", "background_only")


def audit_plan(config: str, python: str = ".\\venv\\Scripts\\python.exe") -> list[dict[str, str]]:
    root = Path("outputs/gate0/background_audit")
    train = []
    for view in VIEWS:
        run = f"resnet18_track_seed407_{view}"
        train.append({"kind": "train", "view": view, "run": run, "command": f'{python} scripts/train_gate0_baseline.py --config {config} --split track --seed 407 --mask-variant {view} --run-name {run}'})
    evaluations = []
    for train_view, eval_view in (("original", "original"), ("foreground_only", "foreground_only"), ("background_only", "background_only"), ("original", "foreground_only"), ("original", "background_only")):
        run = f"resnet18_track_seed407_{train_view}"
        evaluation = f"train_{train_view}__eval_{eval_view}"
        checkpoint = root / run / "best.pt"
        evaluations.append({"kind": "evaluate", "train_view": train_view, "eval_view": eval_view, "command": f'{python} scripts/evaluate_gate0.py --config {config} --checkpoint {checkpoint} --split track --partition test --mask-variant {eval_view} --evaluation-name {evaluation}'})
    return train + evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the fixed Gate-0 background-audit plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=".\\venv\\Scripts\\python.exe")
    args = parser.parse_args()
    print(json.dumps({"seed": 407, "split": "track", "views": VIEWS, "plan": audit_plan(args.config, args.python)}, indent=2))


if __name__ == "__main__":
    main()
