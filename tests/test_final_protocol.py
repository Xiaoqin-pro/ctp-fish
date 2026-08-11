from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_final_protocol_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/validate_final_protocol.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "protocol_valid"' in result.stdout
    assert '"official_test_accessed": false' in result.stdout


def test_outer_manifest_builder_is_deterministic():
    first = subprocess.run(
        [sys.executable, "scripts/build_cxt_fish_outer_manifests.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, "scripts/build_cxt_fish_outer_manifests.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.stdout == second.stdout


def test_outer_training_dry_run_never_loads_outer_test():
    result = subprocess.run(
        [sys.executable, "scripts/train_cxt_fish_outer.py", "--fold", "1", "--method", "F1", "--seed", "3407", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"outer_test_loaded": false' in result.stdout


def test_outer_manifests_have_all_classes_and_disjoint_groups():
    import pandas as pd
    root = ROOT / "outputs" / "cxt_fish" / "final_outer_manifests"
    expected = {str(i) for i in range(1, 17)}
    for fold in ("1", "2", "3"):
        frames = [pd.read_csv(root / f"fold_{fold}" / f"{name}.csv") for name in ("outer_train", "inner_dev", "outer_test")]
        assert all(set(frame.species_id.astype(str)) == expected for frame in frames)
        groups = [set(frame.group_id.astype(str)) for frame in frames]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
