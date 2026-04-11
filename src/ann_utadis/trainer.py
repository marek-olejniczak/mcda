from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .losses import regret_loss


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_acc: list[float]
    train_auc: list[float]
    val_loss: list[float]
    val_acc: list[float]
    val_auc: list[float]


def _batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (logits.detach().cpu().numpy() > 0.0).astype(int)
    y_true = targets.detach().cpu().numpy().astype(int)

    acc = float((preds == y_true).mean())
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, probs))
    return acc, auc


def predict_ann_utadis(model: torch.nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (logits > 0.0).int()
    return preds.cpu().numpy(), probs.cpu().numpy()


def evaluate_ann_utadis(model: torch.nn.Module, dataloader: DataLoader) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for features, targets in dataloader:
            logits = model(features)
            losses.append(float(regret_loss(logits, targets).item()))
            all_logits.append(logits)
            all_targets.append(targets)

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    acc, auc = _batch_metrics(logits_cat, targets_cat)
    return float(np.mean(losses)), acc, auc


def train_ann_utadis(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = 300,
    slope_decrease: bool = True,
    device: str = "cpu",
) -> tuple[torch.nn.Module, TrainingHistory]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs,
    )

    slopes = np.linspace(0.01, 0.003, epochs)
    history = TrainingHistory([], [], [], [], [], [])

    best_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        if slope_decrease:
            model.set_slope(float(slopes[epoch]))

        epoch_losses: list[float] = []
        epoch_logits: list[torch.Tensor] = []
        epoch_targets: list[torch.Tensor] = []

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = regret_loss(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss.item()))
            epoch_logits.append(logits.detach().cpu())
            epoch_targets.append(targets.detach().cpu())

        train_logits = torch.cat(epoch_logits, dim=0)
        train_targets = torch.cat(epoch_targets, dim=0)
        train_acc, train_auc = _batch_metrics(train_logits, train_targets)
        history.train_loss.append(float(np.mean(epoch_losses)))
        history.train_acc.append(train_acc)
        history.train_auc.append(train_auc)

        val_loss, val_acc, val_auc = evaluate_ann_utadis(model, val_loader)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_auc.append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
