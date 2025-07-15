import torch
from ACMPC import CriticTransformer


def test_critic_forward_regression():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            critic = CriticTransformer(2, 1, history_len=3, pred_horizon=2).double()
            critic.eval()

            cs = torch.randn(1, 2, dtype=torch.double)
            ca = torch.randn(1, 1, dtype=torch.double)
            hist = torch.randn(1, 3, 3, dtype=torch.double)

            with torch.no_grad():
                q = critic(cs, ca, hist)
    finally:
        torch.set_default_dtype(prev_dtype)

    expected = torch.tensor([0.1744472288176528], dtype=torch.double)
    assert q.shape == (1,)
    assert torch.allclose(q, expected)
