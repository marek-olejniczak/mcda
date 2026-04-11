from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PreferenceMinMaxScaler:
    gain_columns: list[str]
    cost_columns: list[str]
    min_: dict[str, float] | None = None
    max_: dict[str, float] | None = None

    def fit(self, df: pd.DataFrame) -> "PreferenceMinMaxScaler":
        self.min_ = {}
        self.max_ = {}
        for col in self.gain_columns + self.cost_columns:
            self.min_[col] = float(df[col].min())
            self.max_[col] = float(df[col].max())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler must be fitted before transform().")
        out = df.copy()
        for col in self.gain_columns:
            denom = self.max_[col] - self.min_[col]
            if denom <= 1e-12:
                out[col] = 0.0
            else:
                out[col] = (out[col] - self.min_[col]) / denom
        for col in self.cost_columns:
            denom = self.max_[col] - self.min_[col]
            if denom <= 1e-12:
                out[col] = 1.0
            else:
                out[col] = 1.0 - (out[col] - self.min_[col]) / denom
        return out.clip(0.0, 1.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class NumpyDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.features)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = NumpyDataset(X_train, y_train)
    test_ds = NumpyDataset(X_test, y_test)

    effective_train_batch = len(train_ds) if batch_size is None else batch_size
    effective_test_batch = len(test_ds)

    train_loader = DataLoader(train_ds, batch_size=effective_train_batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=effective_test_batch, shuffle=False)
    return train_loader, test_loader
