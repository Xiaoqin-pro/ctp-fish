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
