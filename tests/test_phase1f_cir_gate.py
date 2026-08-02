from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_phase1f_cir_gate import safe_auc


def test_safe_auc_returns_none_for_single_class():
    assert safe_auc(np.zeros(4, dtype=int), np.arange(4)) is None


def test_safe_auc_is_deterministic_and_finite():
    value = safe_auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert value == 1.0


def test_cir_effect_formula_direction():
    donor_residual = np.array([[0.0, 2.0, -1.0], [1.0, -2.0, 0.5]])
    target = np.array([0, 2])
    donor = np.array([1, 0])
    q = donor_residual[np.arange(2), donor] - donor_residual[np.arange(2), target]
    assert np.allclose(q, [2.0, .5])
