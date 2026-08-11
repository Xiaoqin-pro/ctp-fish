"""Freeze-gate the 90-cell donor sensitivity evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONOR_SEEDS = (4101, 4102, 4103, 4104, 4105)
MODEL_SEEDS = (3407, 2026, 17)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = ROOT / "outputs/cxt_fish/donor_sensitivity/evaluation"
    records = []
    for donor_seed in DONOR_SEEDS:
        for fold in (1, 2, 3):
            manifest = ROOT / f"outputs/cxt_fish/donor_sensitivity/manifests/seed_{donor_seed}/fold_{fold}/outer_context_swap.csv"
            if not manifest.is_file():
                raise SystemExit(f"Missing sensitivity manifest: {manifest}")
            for method in ("F0", "F1"):
                for model_seed in MODEL_SEEDS:
                    path = root / f"seed_{donor_seed}/fold_{fold}/{method}_seed{model_seed}/metrics.json"
                    if not path.is_file(): raise SystemExit(f"Missing sensitivity cell: {path}")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    key = (donor_seed, fold, method, model_seed)
                    if payload.get("official_test_accessed") is not False or payload.get("post_hoc_sensitivity") is not True or payload.get("partition") != "outer_test":
                        raise SystemExit(f"Invalid provenance: {path}")
                    if tuple((payload.get("donor_seed"), payload.get("fold"), payload.get("method"), payload.get("model_seed"))) != key:
                        raise SystemExit(f"Metadata key mismatch: {path}")
                    if payload.get("donor_manifest_sha256") != sha256(manifest):
                        raise SystemExit(f"Manifest SHA mismatch: {path}")
                    primary = ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{fold}/{method}_seed{model_seed}/metrics.json"
                    primary_payload = json.loads(primary.read_text(encoding="utf-8"))
                    if payload.get("checkpoint_sha256") != primary_payload.get("checkpoint_sha256"):
                        raise SystemExit(f"Checkpoint SHA changed: {path}")
                    records.append({"donor_seed": donor_seed, "fold": fold, "method": method, "model_seed": model_seed, "metrics": str(path)})
    if len(records) != 90 or len({(r["donor_seed"], r["fold"], r["method"], r["model_seed"]) for r in records}) != 90:
        raise SystemExit("Expected exactly 90 unique sensitivity cells")
    primary_hashes = {}
    for fold in (1, 2, 3):
        path = ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{fold}/outer_context_swap.csv"
        primary_hashes[str(fold)] = sha256(path)
    output = ROOT / "reports/cxt_fish_donor_sensitivity_complete.json"
    output.write_text(json.dumps({"status": "complete", "cells": 90, "donor_seeds": list(DONOR_SEEDS), "primary_manifest_sha256": primary_hashes, "official_test_accessed": False, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": 90, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
