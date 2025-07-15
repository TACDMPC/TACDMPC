import torch
from ActorCritic import ActorNet


def test_actor_net_output_shapes():
    torch.set_default_dtype(torch.double)
    net = ActorNet(nx=2, nu=1)
    state = torch.randn(4, 2)
    q, r = net(state)
    assert q.shape == (4, 2)
    assert r.shape == (4, 1)

