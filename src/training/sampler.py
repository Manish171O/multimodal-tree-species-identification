"""Class-balanced sampling for the long-tailed species distribution."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_balanced_sampler(labels: np.ndarray, power: float = 0.5, num_samples: int | None = None,
                           seed: int = 0) -> WeightedRandomSampler:
    """Sample each item with probability ∝ n_{y}^{−power}.

    power = 0 -> natural frequencies, 1 -> every class equally likely, 0.5 -> square-root balancing
    (a common compromise that avoids heavily repeating singletons). Seeded for reproducibility.
    """
    labels = np.asarray(labels)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    w = counts[inv].astype(np.float64) ** (-power)                       # (N,)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples or len(labels),
                                 replacement=True, generator=g)
