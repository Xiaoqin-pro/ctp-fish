"""Gate outer evaluation until all nine F0-2RGB cells are complete."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg_path = ROOT / "configs/cxt_fish_rgb2_control_v1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = ROOT / cfg["output_root"]
    expected = []
    for fold in (1, 2, 3):
        for seed in (3407, 2026, 17):
            directory = root / f"fold_{fold}" / f"F0_2RGB_seed{seed}"
            expected.append((fold, seed, directory))
    records = []
    for fold, seed, directory in expected:
        required = [directory / name for name in ("best.pt", "last.pt", "training_curve.csv", "run_metadata.json")]
        if not all(path.is_file() for path in required):
            raise SystemExit(f"F0-2RGB training incomplete: fold={fold} seed={seed}")
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("fold") != fold or metadata.get("seed") != seed or metadata.get("method") != "F0_2RGB":
            raise SystemExit(f"F0-2RGB metadata mismatch: fold={fold} seed={seed}")
        if metadata.get("outer_test_accessed") is not False or metadata.get("official_test_accessed") is not False:
            raise SystemExit(f"F0-2RGB access flag violation: fold={fold} seed={seed}")
        records.append({"fold": fold, "seed": seed, "path": str(directory), "config_sha256": metadata.get("config_sha256")})
    output = ROOT / "reports/cxt_fish_rgb2_training_complete.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "complete", "cells": 9, "official_test_accessed": False, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": 9, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
