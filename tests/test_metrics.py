"""Hand-worked OPA examples (brief T2), cross-checked against scipy kendalltau."""
import torch
from scipy.stats import kendalltau

from src.training.metrics import opa


def test_opa_hand_worked_no_ties():
    y = torch.tensor([10, 20, 15, 40])
    s = torch.tensor([1.0, 3.0, 2.5, 2.0])
    v = opa(s, y)
    assert abs(float(v) - 2 / 3) < 1e-9
    tau = kendalltau(y.numpy(), s.numpy()).statistic
    assert abs(tau - 1 / 3) < 1e-9
    assert abs(float(v) - (tau + 1) / 2) < 1e-9


def test_opa_hand_worked_with_tie():
    y = torch.tensor([10, 20, 15, 20])
    s = torch.tensor([1.0, 3.0, 2.5, 2.0])
    v = opa(s, y)
    assert abs(float(v) - 0.8) < 1e-9


def test_opa_perfect_and_reversed():
    y = torch.tensor([1, 2, 3, 4])
    assert float(opa(torch.tensor([1.0, 2.0, 3.0, 4.0]), y)) == 1.0
    assert float(opa(torch.tensor([4.0, 3.0, 2.0, 1.0]), y)) == 0.0


def test_opa_all_tied_returns_nan():
    y = torch.tensor([5, 5, 5])
    v = opa(torch.tensor([1.0, 2.0, 3.0]), y)
    assert torch.isnan(v)
