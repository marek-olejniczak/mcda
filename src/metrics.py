from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, float]:
	"""Return core binary-classification metrics rounded to 4 decimals.

	Args:
		y_true: Ground-truth labels (0/1).
		y_pred: Predicted class labels (0/1).
		y_prob: Predicted probability for class 1.
	"""
	y_prob = np.asarray(y_prob)
	if y_prob.ndim > 1:
		y_prob = y_prob[:, 1]

	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"f1": f1_score(y_true, y_pred),
		"auc": roc_auc_score(y_true, y_prob),
	}
	return {name: round(float(value), 4) for name, value in metrics.items()}