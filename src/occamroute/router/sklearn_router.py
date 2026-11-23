"""Calibrated Scikit-learn Gradient Boosting complexity router."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from occamroute.feature_extractor import FEATURE_NAMES


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error across uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


@dataclass(frozen=True)
class SklearnRouterConfig:
    """Hyperparameters for the calibrated GradientBoosting router."""

    n_estimators: int = 120
    learning_rate: float = 0.08
    max_depth: int = 3
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: int = 3
    random_state: int = 42


class SklearnComplexityRouter:
    """StandardScaler + GradientBoosting + CalibratedClassifierCV router persisted via joblib."""

    def __init__(self, config: SklearnRouterConfig | None = None) -> None:
        self.config = config or SklearnRouterConfig()
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        base_clf = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=base_clf,
                        method=self.config.calibration_method,
                        cv=self.config.cv,
                    ),
                ),
            ]
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnComplexityRouter:
        self.pipeline.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SklearnComplexityRouter must be fitted before calling predict_proba.")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        probs = self.pipeline.predict_proba(arr)[:, 1]
        return np.asarray(probs, dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
        y_true = np.asarray(y, dtype=np.int64)
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(np.int64)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "ece": compute_ece(y_true, y_prob),
        }

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, out)
        return out

    @classmethod
    def load(cls, path: Path | str) -> SklearnComplexityRouter:
        payload = joblib.load(Path(path))
        obj = cls(config=SklearnRouterConfig(**payload["config"]))
        obj.feature_names = tuple(payload["feature_names"])
        obj.pipeline = payload["pipeline"]
        obj.is_fitted = bool(payload["is_fitted"])
        return obj
