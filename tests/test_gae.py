import torch
from ACMPC.training_loop import compute_gae_and_returns


def test_gae_matches_returns_when_no_dones():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.zeros(3, dtype=torch.bool)
    next_value = torch.tensor(0.0)
    adv, ret = compute_gae_and_returns(rewards, values, dones, next_value, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([3.0, 2.0, 1.0]))
    assert torch.allclose(ret, torch.tensor([3.0, 2.0, 1.0]))


def test_gae_with_terminal_state():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    dones = torch.tensor([0, 0, 1], dtype=torch.bool)
    next_value = torch.tensor(0.0)
    adv, ret = compute_gae_and_returns(rewards, values, dones, next_value)
    expected_adv = torch.tensor([5.1540, 3.8958, 1.5000])
    expected_ret = torch.tensor([5.6540, 4.8958, 3.0000])
    assert torch.allclose(adv, expected_adv, atol=1e-4)
    assert torch.allclose(ret, expected_ret, atol=1e-4)
