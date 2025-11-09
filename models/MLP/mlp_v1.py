"""
MLP Agent for Permuted MNIST
Trainable agent with a compact, robust MLP; respects the site template and 60s/task budget.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import time
import numpy as np

import torch
from torch import nn

# Limiter les threads pour éviter les timeouts/oversubscription
torch.set_num_threads(2)

# ----------------------------
# Utils: MLP builder
# ----------------------------
_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    None: nn.Identity,
}

def _build_mlp(
    input_dim: int = 28 * 28,
    hidden_layers=(1536, 1024, 512),
    activation: str = "relu",
    norm_type: str = "none",       # "none" | "layernorm"
    dropout: float = 0.10,
    output_dim: int = 10,
) -> nn.Sequential:
    layers, d = [], input_dim
    Act = _ACTS[activation]
    for h in hidden_layers:
        layers.append(nn.Linear(d, h))
        if norm_type == "layernorm":
            layers.append(nn.LayerNorm(h))
        layers.append(Act())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


class Agent:
    """Trainable MLP agent compatible with the platform template."""

    def __init__(self, output_dim: int = 10, seed: int = 42):
        """
        Args:
            output_dim: Number of output classes (10 for MNIST digits)
            seed: Random seed for reproducibility
        """
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)

        # Seed PyTorch + NumPy
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Config robuste (bon ratio perf/temps)
        self.input_dim = 28 * 28
        self.hidden_layers = (1536, 1024, 512)
        self.activation = "relu"
        self.norm_type = "none"     # "none" ou "layernorm"
        self.dropout = 0.10

        # Opti stable
        self.optimizer_name = "adamw"
        self.lr = 1.3e-3
        self.weight_decay = 1e-4

        # Entraînement
        self.batch_size = 2048
        self.max_epochs = 10
        self.val_fraction = 0.10
        self.time_budget_s = 60.0

        # Construction du modèle et de la loss
        self.model = _build_mlp(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            dropout=self.dropout,
            output_dim=self.output_dim,
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ----------------------------
    # API demandée par la plateforme
    # ----------------------------
    def reset(self):
        """Reset the agent for a new task/simulation (re-initialize weights)."""
        self.model = _build_mlp(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            norm_type=self.norm_type,
            dropout=self.dropout,
            output_dim=self.output_dim,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the agent on the provided data.

        Args:
            X_train: (N, 28, 28) or (N, 784)
            y_train: (N,) or (N, 1)
        """
        # Prétraitement
        X = self._to_tensor(X_train)
        y = self._to_labels(y_train)

        # Split simple train/val
        N = X.size(0)
        idx = torch.randperm(N)
        N_val = int(self.val_fraction * N)
        val_idx, tr_idx = idx[:N_val], idx[N_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Optimizer
        if self.optimizer_name.lower() == "adamw":
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Entraînement avec budget-temps
        start = time.time()
        self.model.train()

        bs = self.batch_size
        for _ in range(self.max_epochs):
            if time.time() - start > self.time_budget_s:
                break
            perm = torch.randperm(X_tr.size(0))
            for i in range(0, X_tr.size(0), bs):
                if time.time() - start > self.time_budget_s:
                    break
                b = perm[i:i + bs]
                xb, yb = X_tr[b], y_tr[b]

                opt.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                opt.step()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the test set.

        Args:
            X_test: (N, 28, 28) or (N, 784)

        Returns:
            Predicted class labels (N,)
        """
        X = self._to_tensor(X_test)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    # ----------------------------
    # Helpers
    # ----------------------------
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        # Accept (N, 28, 28) or (N, 784)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = torch.from_numpy(X).float().div_(255.0)
        return X

    def _to_labels(self, y: np.ndarray) -> torch.Tensor:
        if y.ndim > 1:
            y = y.squeeze()
        return torch.from_numpy(y).long()