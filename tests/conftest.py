import torch
import pytest

@pytest.fixture(scope="session")
def device():
    """Restituisce cuda se disponibile, altrimenti cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
