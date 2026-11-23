"""Lightweight PyTorch tabular MLP complexity router."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from occamroute.feature_extractor import FEATURE_NAMES
from occamroute.router.sklearn_router import compute_ece


@dataclass(frozen=True)
class TorchRouterConfig:
    """Hyperparameters for the PyTorch tabular MLP router."""

    input_dim: int = len(FEATURE_NAMES)
    hidden_dims: tuple[int, int] = (32, 16)
    dropout: float = 0.15
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 120
    batch_size: int = 32
    seed: int = 42


class TabularRouterMLP(nn.Module):
    """Compact MLP with LayerNorm, SiLU, and Dropout for tabular graph features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (32, 16),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchComplexityRouter:
    """PyTorch MLP router with built-in z-score normalization and `.pt` serialization."""

    def __init__(self, config: TorchRouterConfig | None = None) -> None:
        self.config = config or TorchRouterConfig()
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        torch.manual_seed(self.config.seed)
        self.device = torch.device("cpu")
        self.model = TabularRouterMLP(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)
        self.mean_: np.ndarray = np.zeros(self.config.input_dim, dtype=np.float32)
        self.std_: np.ndarray = np.ones(self.config.input_dim, dtype=np.float32)
        self.train_losses: list[float] = []
        self.is_fitted: bool = False

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return (arr - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchComplexityRouter:
        torch.manual_seed(self.config.seed)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.mean_ = X_arr.mean(axis=0).astype(np.float32)
        self.std_ = np.maximum(X_arr.std(axis=0), 1e-5).astype(np.float32)
        X_norm = self._normalize(X_arr)

        ds = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y_arr))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        criterion = nn.BCEWithLogitsLoss()

        self.train_losses = []
        self.model.train()
        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(xb)
            scheduler.step()
            self.train_losses.append(epoch_loss / max(1, len(X_arr)))

        self.model.eval()
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TorchComplexityRouter must be fitted before calling predict_proba.")
        X_norm = self._normalize(X)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(X_norm).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.asarray(probs, dtype=np.float32).reshape(-1)

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
        cfg_dict = asdict(self.config)
        cfg_dict["hidden_dims"] = list(self.config.hidden_dims)
        checkpoint = {
            "config": cfg_dict,
            "feature_names": list(self.feature_names),
            "state_dict": self.model.state_dict(),
            "mean": torch.from_numpy(self.mean_),
            "std": torch.from_numpy(self.std_),
            "is_fitted": self.is_fitted,
        }
        torch.save(checkpoint, out)
        return out

    @classmethod
    def load(cls, path: Path | str) -> TorchComplexityRouter:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        cfg_raw = dict(ckpt["config"])
        cfg_raw["hidden_dims"] = tuple(cfg_raw["hidden_dims"])
        obj = cls(config=TorchRouterConfig(**cfg_raw))
        obj.feature_names = tuple(ckpt["feature_names"])
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        obj.mean_ = ckpt["mean"].cpu().numpy().astype(np.float32)
        obj.std_ = ckpt["std"].cpu().numpy().astype(np.float32)
        obj.is_fitted = bool(ckpt["is_fitted"])
        return obj
