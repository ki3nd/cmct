
import torch

from cmct.branch_lora.lora.apply import compute_rank
from tests.conftest import load_fixture


def test_rank_ramp_matches_the_frozen_baseline():
    fx = load_fixture("rank_ramp.json")
    got = [compute_rank(i, fx["r"], fx["ramp"]) for i in range(12)]
    assert got == fx["expected"]


def test_rank_ramp_disabled_is_flat():
    assert [compute_rank(i, 2, []) for i in range(12)] == [2] * 12


def test_ema_update_is_convex_combination():
    from cmct.branch_lora.ema import ema_update_lora_params

    class M(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.full((2, 2), value))

    ema, src = M(0.0), M(1.0)
    ema_update_lora_params(ema, src, 0.99)
    assert torch.allclose(ema.lora_A, torch.full((2, 2), 0.01))
