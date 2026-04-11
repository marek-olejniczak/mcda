from __future__ import annotations

import torch
import torch.nn.functional as F


def regret_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    return torch.mean(F.relu(-target * output) + F.relu((1.0 - target) * output))
