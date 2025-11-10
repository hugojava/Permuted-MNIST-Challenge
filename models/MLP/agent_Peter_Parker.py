# agent.py
"""
Permuted MNIST — Submission Agent (CPU-only, <60s/task)
API compatible template:
  - __init__(output_dim=10, seed=None)
  - reset()
  - train(X_train, y_train)
  - predict(X_test)
Config = (2048, 1024) + GELU(approx) + LayerNorm + Dropout 0.10
Batch = 2048, lr = 1e-3, AdamW, label_smoothing = 0.05
"""

import os
import time
import numpy as np

# ---- Limitation threads / CPU-only ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import torch
from torch import nn

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def build_mlp(
    input_dim: int = 784,
    hidden=(2048, 1024),          # <- best config
    activation: str = "gelu",     # "gelu" | "relu" | "tanh"
    norm: str = "layernorm",      # "layernorm" | "none"
    dropout: float = 0.10,
    num_classes: int = 10,
    gelu_approx: bool = True,
) -> nn.Sequential:
    if activation == "gelu":
        Act = (lambda: nn.GELU(approximate="tanh")) if gelu_approx else nn.GELU
    elif activation == "relu":
        Act = nn.ReLU
    elif activation == "tanh":
        Act = nn.Tanh
    else:
        raise ValueError(f"Unknown activation: {activation}")

    layers = []
    d = input_dim
    for h in hidden:
        use_bias = (norm == "none")  # si norme affine derrière, bias peu utile
        layers.append(nn.Linear(d, h, bias=use_bias))
        if norm == "layernorm":
            layers.append(nn.LayerNorm(h, elementwise_affine=True))
        layers.append(Act())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, num_classes, bias=True))

    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


class Agent:
    """MLP agent compatible avec le template de la plateforme."""

    def __init__(self, output_dim: int = 10, seed: int | None = None):
        # Hyperparamètres = ta meilleure config validée
        self.input_dim = 28 * 28
        self.num_classes = int(output_dim)
        self.hidden = (2048, 1024)          # <- best
        self.activation = "gelu"
        self.norm = "layernorm"
        self.dropout = 0.10

        self.batch_size = 2048              # <- best
        self.max_epochs = 10
        self.val_ratio = 0.10
        self.lr = 1.0e-3                    # <- best
        self.weight_decay = 1e-4
        self.label_smoothing = 0.05

        # Budget temps par task (60 s plateforme) — marge
        self.time_budget_s = 58.0

        # Seeds (compatibles template)
        if seed is None:
            seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        # CPU only
        self.device = torch.device("cpu")

        # Modèle + loss
        self.model = build_mlp(
            input_dim=self.input_dim,
            hidden=self.hidden,
            activation=self.activation,
            norm=self.norm,
            dropout=self.dropout,
            num_classes=self.num_classes,
            gelu_approx=True,
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # --- API requise par la plateforme ---
    def reset(self):
        """Ré-initialise le réseau (nouvelle task/simulation)."""
        self.model = build_mlp(
            input_dim=self.input_dim,
            hidden=self.hidden,
            activation=self.activation,
            norm=self.norm,
            dropout=self.dropout,
            num_classes=self.num_classes,
            gelu_approx=True,
        ).to(self.device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le modèle en respectant le time budget.
        X_train: (N, 28, 28) ou (N, 784), uint8 ou float
        y_train: (N,) ou (N,1)
        """
        X = self._to_tensor(X_train).to(self.device)  # float32, [0,1]
        y = self._to_labels(y_train).to(self.device)  # int64

        # Split train/val
        N = X.size(0)
        idx = torch.randperm(N)
        n_val = int(self.val_ratio * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        eval_every = 2       # validation 1 epoch sur 2
        patience = 4         # early stopping
        best_acc = 0.0
        best_state = None
        start = time.time()

        for ep in range(self.max_epochs):
            if time.time() - start > self.time_budget_s:
                break

            self.model.train()
            perm = torch.randperm(X_tr.size(0))
            for i in range(0, X_tr.size(0), self.batch_size):
                if time.time() - start > self.time_budget_s:
                    break
                b = perm[i:i + self.batch_size]
                xb, yb = X_tr[b], y_tr[b]

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                opt.step()

            # Validation espacée + early stopping
            do_val = ((ep + 1) % eval_every == 0) or (ep + 1 == self.max_epochs)
            if do_val:
                self.model.eval()
                with torch.no_grad():
                    preds_val = self.model(X_val).argmax(1)
                    val_acc = (preds_val == y_val).float().mean().item()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 4
                else:
                    patience -= 1
                    if patience == 0:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Retourne un np.ndarray (N,) des classes prédites [0..9]."""
        X = self._to_tensor(X_test).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            preds = logits.argmax(1).cpu().numpy()
        return preds

    # --- Helpers ---
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        # Accepte (N, 28, 28) ou (N, 784), uint8 ou float
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = torch.from_numpy(X).float()
        if X.numel() > 0 and X.max() > 1.0 + 1e-6:  # cas [0..255]
            X = X / 255.0
        return X

    def _to_labels(self, y: np.ndarray) -> torch.Tensor:
        if y.ndim > 1:
            y = y.squeeze()
        return torch.from_numpy(y).long()