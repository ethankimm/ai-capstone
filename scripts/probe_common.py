"""Model-agnostic ridge/MLP probe-fitting primitives, shared by `probe_codi.py` and
`probe_coconut.py`. No sklearn/scipy dependency: ridge is closed-form numpy-free torch,
the MLP is a small torch module trained with Adam. Target transform is signed-log1p to
compress GSM8K's value range before regressing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def parse_value(v: str) -> Optional[float]:
    v = v.strip().lstrip("+")
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def inv_signed_log1p(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(y.abs())


def within_tol(pred: float, gold: float, tol: float) -> bool:
    denom = max(1.0, abs(gold))
    return abs(pred - gold) / denom <= tol


def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge: w = (X^T X + lam*I)^-1 X^T y, X already has a bias column."""
    d = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    Xty = X.T @ y
    return torch.linalg.solve(XtX, Xty)


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X: torch.Tensor, y: torch.Tensor, hidden: int, epochs: int, lr: float) -> TinyMLP:
    mlp = TinyMLP(X.shape[1], hidden)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = mlp(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return mlp
