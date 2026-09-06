"""Familia MLP: red densa sobre las mismas 17 features que el resto.

No es un caso especial en ningún punto del banco: entra al mismo Pipeline,
a los mismos folds y a las mismas métricas que XGBoost o SVR, de modo que la
comparación es 1:1.

Wrapper propio en vez de skorch: evita una dependencia nueva para envolver
un modelo de tres capas, y deja explícito el control de semilla y de épocas,
que es lo que hace reproducible —y por tanto comparable— este candidato.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ....config import ML_RANDOM_STATE

FAMILY = "mlp"
SCALE_FEATURES = True

_DEFAULTS = {
    "hidden_sizes": (64, 32),
    "epochs": 200,
    "learning_rate": 0.001,
    "batch_size": 256,
    "dropout": 0.1,
}


def _build_network(n_features: int, hidden_sizes, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = n_features
    for size in hidden_sizes:
        layers.extend([nn.Linear(in_features, size), nn.ReLU(), nn.Dropout(dropout)])
        in_features = size
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class _TorchMLPBase(BaseEstimator):
    """Parte común de clasificador y regresor: red, bucle de ajuste, semilla."""

    def __init__(
        self,
        hidden_sizes=(64, 32),
        epochs: int = 200,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        dropout: float = 0.1,
        random_state: int = ML_RANDOM_STATE,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def _fit_network(self, X, y, loss_fn) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)

        self.network_ = _build_network(
            X_tensor.shape[1], tuple(self.hidden_sizes), self.dropout
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)

        n_samples = len(X_tensor)
        generator = torch.Generator().manual_seed(self.random_state)
        self.network_.train()
        for _ in range(self.epochs):
            order = torch.randperm(n_samples, generator=generator)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.network_(X_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimizer.step()
        self.network_.eval()

    def _forward(self, X) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(X, dtype=np.float32))
            return self.network_(tensor).numpy().ravel()


class TorchMLPClassifier(ClassifierMixin, _TorchMLPBase):
    """MLP binario. El desbalance entra como ``pos_weight`` de la pérdida."""

    def __init__(
        self,
        hidden_sizes=(64, 32),
        epochs: int = 200,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        dropout: float = 0.1,
        random_state: int = ML_RANDOM_STATE,
        scale_pos_weight: float = 1.0,
    ):
        # Parametros explicitos, no **kwargs: BaseEstimator.get_params sólo
        # introspecciona la firma de __init__ de la clase concreta (no
        # recorre el MRO), así que un **kwargs aquí haría desaparecer
        # hidden_sizes/epochs/... de get_params() y rompería clone().
        super().__init__(
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dropout=dropout,
            random_state=random_state,
        )
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(self.scale_pos_weight)])
        )
        self._fit_network(X, y, loss_fn)
        return self

    def predict_proba(self, X) -> np.ndarray:
        positive = 1.0 / (1.0 + np.exp(-self._forward(X)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchMLPRegressor(RegressorMixin, _TorchMLPBase):
    """MLP de regresión sobre el target ya transformado con log1p."""

    def fit(self, X, y):
        self._fit_network(X, y, nn.MSELoss())
        return self

    def predict(self, X) -> np.ndarray:
        return self._forward(X)


def _settings(params: dict) -> dict:
    allowed = set(_DEFAULTS)
    return {**_DEFAULTS, **{k: v for k, v in params.items() if k in allowed}}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    model = TorchMLPClassifier(scale_pos_weight=float(scale_pos_weight), **_settings(params))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    model = TorchMLPRegressor(**_settings(params))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
