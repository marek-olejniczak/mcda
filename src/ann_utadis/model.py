from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MonotonicLayer, ThresholdLayer


class UtaModel(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.monotonic_layer = MonotonicLayer(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
            slope=slope,
        )

    def set_slope(self, value: float) -> None:
        self.monotonic_layer.set_slope(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marginal_values = self.monotonic_layer(x)
        return marginal_values.sum(dim=1)


class NormLayer(nn.Module):
    def __init__(self, utility_model: UtaModel, num_criteria: int) -> None:
        super().__init__()
        self.utility_model = utility_model
        self.num_criteria = num_criteria

    def set_slope(self, value: float) -> None:
        self.utility_model.set_slope(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        utility = self.utility_model(x)
        device = utility.device
        zero_input = torch.zeros((1, self.num_criteria), device=device)
        one_input = torch.ones((1, self.num_criteria), device=device)
        zero_utility = self.utility_model(zero_input)
        one_utility = self.utility_model(one_input)
        denom = torch.clamp(one_utility - zero_utility, min=1e-8)
        return (utility - zero_utility) / denom


class AnnUtadis(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int = 30,
        slope: float = 0.01,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        self.num_hidden_components = num_hidden_components
        self.utility_model = UtaModel(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
            slope=slope,
        )
        self.norm_layer = NormLayer(self.utility_model, num_criteria=num_criteria)
        self.threshold_layer = ThresholdLayer(threshold=threshold)

    def set_slope(self, value: float) -> None:
        self.norm_layer.set_slope(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.norm_layer(x)
        return self.threshold_layer(score)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_layer(x)
