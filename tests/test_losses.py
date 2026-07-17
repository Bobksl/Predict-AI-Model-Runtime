"""Loss sanity checks (brief T3)."""
import math
import torch

from src.training.losses import pairwise_hinge_loss, listmle_loss


def test_hinge_zero_on_perfect_order_beyond_margin():
    y = torch.tensor([[1, 2, 3, 4]])
    s = torch.tensor([[0.0, 10.0, 20.0, 30.0]])  # huge margin, perfectly ordered
    loss = pairwise_hinge_loss(s, y, margin=0.1)
    assert float(loss) == 0.0


def test_hinge_positive_on_reversed_order():
    y = torch.tensor([[1, 2]])
    s = torch.tensor([[10.0, 0.0]])  # wrong way round
    loss = pairwise_hinge_loss(s, y, margin=0.1)
    assert float(loss) > 0


def test_listmle_matches_hand_rolled_three_item():
    # y descending order is already [item2, item1, item0] (values 30,20,10)
    y = torch.tensor([[10, 20, 30]])
    s = torch.tensor([[1.0, 2.0, 3.0]])
    loss = listmle_loss(s, y)
    # pi* = sort by y descending -> indices [2,1,0] -> scores [3,2,1]
    ss = torch.tensor([3.0, 2.0, 1.0])
    expected = 0.0
    # NLL = sum_i [ logsumexp(s_i:) - s_i ]
    for i in range(3):
        tail = ss[i:]
        expected += torch.logsumexp(tail, dim=0).item() - ss[i].item()
    assert math.isclose(float(loss), expected, rel_tol=1e-5)


def test_listmle_finite_at_extreme_scores():
    y = torch.tensor([[1, 2, 3]])
    s = torch.tensor([[1e6, -1e6, 0.0]])
    loss = listmle_loss(s, y)
    assert torch.isfinite(loss)
