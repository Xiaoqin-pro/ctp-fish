"""Generate the no-training reviewer-risk audits from frozen artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "outputs" / "gate0" / "audit"
MANIFEST_ROOT = ROOT / "outputs" / "cxt_fish" / "final_outer_manifests"


def _sid(value) -> str:
    text = str(value)
    return text.zfill(2) if text.isdigit() else text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def class_inclusion_audit() -> None:
    source = AUDIT_ROOT / "f4k16t_classes.csv"
    frame = pd.read_csv(source, dtype={"species_id": str})
    frame["species_id"] = frame["species_id"].map(_sid)
    frame["num_groups"] = frame["num_tracks"].astype(int)
    frame["inclusion_threshold"] = 15
    frame["included"] = frame["num_groups"] >= 15
    frame["exclusion_reason"] = frame.apply(
        lambda row: "" if row["included"] else "fewer_than_15_recorded_groups", axis=1
    )
    if len(frame) != 23 or int(frame["included"].sum()) != 16:
        raise AssertionError("Frozen F4K-16T class inclusion is not 23 -> 16.")
    if int(frame["num_images"].sum()) != 27370:
        raise AssertionError("Unexpected frozen image total.")
    if int(frame.loc[frame["included"], "num_images"].sum()) != 27133:
        raise AssertionError("Unexpected retained image total.")
    if (frame.loc[frame["included"], "num_groups"] < 15).any():
        raise AssertionError("Included class below the frozen group threshold.")
    out = frame[[
        "species_id", "species_name", "num_images", "num_groups", "included",
        "inclusion_threshold", "exclusion_reason",
    ]]
    out.to_csv(ROOT / "experiments" / "f4k23_class_inclusion_audit.csv", index=False)
    rows = [
        "# F4K-16T class inclusion audit",
        "",
        "The frozen Gate-0 rule retains species with at least 15 recorded groups.",
        "The historical `num_tracks` field is reported here as recorded-group /",
        "trajectory-proxy count. The threshold was pre-specified before method",
        "development; this report does not claim it is theoretically required for",
        "three-fold evaluation.",
        "",
        f"- All species: {len(frame)}",
        f"- Retained species: {int(frame['included'].sum())}",
        f"- Excluded species: {int((~frame['included']).sum())}",
        f"- All images: {int(frame['num_images'].sum())}",
        f"- Retained images: {int(frame.loc[frame['included'], 'num_images'].sum())}",
        f"- Excluded images: {int(frame.loc[~frame['included'], 'num_images'].sum())}",
        "",
        "| Species | Images | Recorded groups | Included | Exclusion reason |",
        "|---|---:|---:|:---:|---|",
    ]
    for row in out.itertuples(index=False):
        rows.append(
            f"| {row.species_id} {row.species_name} | {row.num_images} | "
            f"{row.num_groups} | {'yes' if row.included else 'no'} | "
            f"{row.exclusion_reason or '—'} |"
        )
    _write(ROOT / "reports" / "f4k16t_class_inclusion_audit.md", "\n".join(rows) + "\n")


def donor_audit() -> None:
    distribution_rows: list[dict] = []
    group_rows: list[dict] = []
    matrix_rows: list[dict] = []
    report = [
        "# Frozen donor protocol audit",
        "",
        "This audit describes, but does not modify, the primary seed-3407",
        "context-swap manifests. Each recipient has one same-class and one",
        "cross-class donor from the same held-out outer fold. Donor candidates",
        "are ordered deterministically and selected by the frozen hash rule.",
        "The cross-class pool is not species-balanced before selection; its",
        "frequency is therefore reported rather than hidden.",
        "",
    ]
    for fold in (1, 2, 3):
        fold_root = MANIFEST_ROOT / f"fold_{fold}"
        manifest_path = fold_root / "outer_context_swap.csv"
        outer_path = fold_root / "outer_test.csv"
        manifest = pd.read_csv(manifest_path, dtype=str)
        outer = pd.read_csv(outer_path, dtype=str)
        outer["species_id"] = outer["species_id"].map(_sid)
        manifest["recipient_species_id"] = manifest["recipient_species_id"].map(_sid)
        manifest["donor_species_id"] = manifest["donor_species_id"].map(_sid)
        if len(manifest) != outer["image_path"].nunique() * 2:
            raise AssertionError(f"fold {fold}: expected two donor rows per recipient")
        if set(manifest["swap_type"]) != {"same_class_cross_track", "cross_class"}:
            raise AssertionError(f"fold {fold}: unexpected swap types")
        outer_paths = set(outer["image_path"])
        for recipient, rows in manifest.groupby("recipient_image_path", sort=False):
            if len(rows) != 2 or set(rows["swap_type"]) != {"same_class_cross_track", "cross_class"}:
                raise AssertionError(f"fold {fold}: recipient does not have exactly two rows: {recipient}")
            for row in rows.itertuples(index=False):
                if str(row.supported).lower() != "true":
                    raise AssertionError(f"fold {fold}: unsupported primary donor")
                if row.donor_image_path not in outer_paths:
                    raise AssertionError(f"fold {fold}: donor outside outer-test")
                if row.donor_group_id == row.recipient_group_id:
                    raise AssertionError(f"fold {fold}: donor group overlaps recipient")
                if row.swap_type == "same_class_cross_track" and row.donor_species_id != row.recipient_species_id:
                    raise AssertionError(f"fold {fold}: same-class donor mismatch")
                if row.swap_type == "cross_class" and row.donor_species_id == row.recipient_species_id:
                    raise AssertionError(f"fold {fold}: cross-class donor mismatch")
        cross = manifest[manifest["swap_type"] == "cross_class"]
        species_counts = cross["donor_species_id"].value_counts().sort_index()
        total = int(species_counts.sum())
        for species, count in species_counts.items():
            distribution_rows.append({"fold": fold, "swap_type": "cross_class", "donor_species_id": species,
                                      "count": int(count), "fraction": float(count / total)})
        for (species, group), count in cross.groupby(["donor_species_id", "donor_group_id"]).size().items():
            group_rows.append({"fold": fold, "swap_type": "cross_class", "donor_species_id": species,
                               "donor_group_id": group, "count": int(count)})
        matrix = cross.groupby(["recipient_species_id", "donor_species_id"]).size().reset_index(name="count")
        matrix["fold"] = fold
        matrix_rows.extend(matrix.to_dict("records"))
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report.extend([
            f"## Fold {fold}",
            "",
            f"- Rows: {len(manifest)}; recipients: {manifest['recipient_image_path'].nunique()}",
            f"- Manifest SHA-256: `{digest}`",
            f"- Cross-class donor species: {len(species_counts)}",
            f"- Cross-class donor frequency range: {species_counts.min()/total:.4f}--{species_counts.max()/total:.4f}",
            f"- Maximum cross-class donor-image reuse: {cross['donor_image_path'].value_counts().max()}",
            f"- Maximum cross-class donor-group reuse: {cross['donor_group_id'].value_counts().max()}",
            "",
        ])
    pd.DataFrame(distribution_rows).to_csv(ROOT / "experiments" / "cxt_fish_primary_donor_distribution.csv", index=False)
    pd.DataFrame(group_rows).to_csv(ROOT / "experiments" / "cxt_fish_primary_donor_group_usage.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(ROOT / "experiments" / "cxt_fish_primary_recipient_donor_matrix.csv", index=False)
    report.extend([
        "## Fixed semantics",
        "",
        "The manifests are shared by F0 and F1 within each fold. Donor selection",
        "uses the frozen seed 3407 and does not first balance donor species. The",
        "composites therefore represent controlled conflicting donor-context",
        "interventions, not guaranteed fish-free background replacement.",
    ])
    _write(ROOT / "reports" / "cxt_fish_donor_protocol_audit.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    class_inclusion_audit()
    donor_audit()
    print("Reviewer audits generated from frozen artifacts.")
