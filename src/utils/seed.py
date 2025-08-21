"""
src/utils/seed.py
Utility for deterministic runs
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    print(f"[set_seed] Using seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
