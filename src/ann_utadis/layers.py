from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyHardSigmoid(nn.Module):
    def __init__(self, slope: float = 0.01) -> None:
        super().__init__()
        self.slope = slope

    def set_slope(self, value: float) -> None:
        self.slope = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(1.0 - F.leaky_relu(1 - x, self.slope), self.slope)


class CriterionLayerSpread(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        input_range: tuple[float, float] = (0.0, 1.0),
        normalize_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        self.normalize_bias = normalize_bias
        self.min_weight = 0.0

        inverted_range = (-input_range[0], -input_range[1])
        self.min_bias = min(inverted_range)
        self.max_bias = max(inverted_range)

        self.bias = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 1.0, 10.0)
        nn.init.uniform_(self.bias, self.min_bias, self.max_bias)

    def compute_bias(self) -> torch.Tensor:
        if self.normalize_bias:
            return torch.clamp(self.bias, self.min_bias, self.max_bias)
        return self.bias

    def compute_weight(self) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data[self.weight.data < 0] = self.min_weight
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, criteria), got shape {tuple(x.shape)}")
        spread_x = x.view(-1, 1, self.num_criteria)
        return (spread_x + self.compute_bias()) * self.compute_weight()


class CriterionLayerCombine(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        min_weight: float = 0.001,
    ) -> None:
        super().__init__()
        self.min_weight = min_weight
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 0.2, 1.0)
        self.weight.data = self.weight.data / torch.sum(self.weight.data)

    def compute_weight(self) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data[self.weight.data < 0] = self.min_weight
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch, hidden_components, criteria), got shape {tuple(x.shape)}"
            )
        return (x * self.compute_weight()).sum(dim=1)


class MonotonicLayer(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.criterion_layer_spread = CriterionLayerSpread(num_criteria, num_hidden_components)
        self.activation_function = LeakyHardSigmoid(slope=slope)
        self.criterion_layer_combine = CriterionLayerCombine(num_criteria, num_hidden_components)

    def set_slope(self, value: float) -> None:
        self.activation_function.set_slope(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.criterion_layer_spread(x)
        x = self.activation_function(x)
        x = self.criterion_layer_combine(x)
        return x


class ThresholdLayer(nn.Module):
    def __init__(self, threshold: float = 0.5, requires_grad: bool = True) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([threshold], dtype=torch.float32), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.threshold
