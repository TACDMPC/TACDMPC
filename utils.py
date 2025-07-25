import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

