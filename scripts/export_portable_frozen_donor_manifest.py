"""Export frozen donor pairings without machine-specific absolute paths.

This script is read-only with respect to the frozen manifests.  It records the
exact pairings using normalized dataset-relative identifiers and stable hashes,
so a local data copy can resolve the rows without relying on D:\\datasets or
any other absolute root.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\datasets\Fish4Knowledge")
OUT = ROOT / "reports" / "portable_frozen_donor_manifest.csv"


def normalized_relative(path: str) -> str:
    raw = str(path).replace("/", "\\")
    prefix = str(DATA_ROOT).replace("/", "\\").rstrip("\\") + "\\"
    if not raw.lower().startswith(prefix.lower()):
        raise ValueError(f"path is outside the declared dataset root: {path}")
    return raw[len(prefix):].replace("\\", "/")


def stable_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def main() -> None:
    rows: list[pd.DataFrame] = []
    primary = ROOT / "outputs" / "cxt_fish" / "final_outer_manifests"
    rows.append(pd.concat([
        pd.read_csv(primary / f"fold_{fold}" / "outer_context_swap.csv").assign(
            fold=fold, donor_realization="primary_3407"
        ) for fold in (1, 2, 3)
    ], ignore_index=True))
    sensitivity = ROOT / "outputs" / "cxt_fish" / "donor_sensitivity" / "manifests"
    for seed in (4101, 4102, 4103, 4104, 4105):
        rows.append(pd.concat([
            pd.read_csv(sensitivity / f"seed_{seed}" / f"fold_{fold}" / "outer_context_swap.csv").assign(
                fold=fold, donor_realization=f"seed_{seed}"
            ) for fold in (1, 2, 3)
        ], ignore_index=True))
    frame = pd.concat(rows, ignore_index=True)
    frame["recipient_relative_id"] = frame["recipient_image_path"].map(normalized_relative)
    frame["donor_relative_id"] = frame["donor_image_path"].map(normalized_relative)
    frame["recipient_stable_id"] = frame["recipient_relative_id"].map(stable_id)
    frame["donor_stable_id"] = frame["donor_relative_id"].map(stable_id)
    frame = frame[
        [
            "donor_realization", "fold", "seed", "swap_type", "supported",
            "recipient_stable_id", "recipient_relative_id", "recipient_group_id", "recipient_species_id",
            "donor_stable_id", "donor_relative_id", "donor_group_id", "donor_species_id",
        ]
    ].sort_values(["donor_realization", "fold", "recipient_relative_id", "swap_type"], kind="stable")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {len(frame)} rows to {OUT}")


if __name__ == "__main__":
    main()
