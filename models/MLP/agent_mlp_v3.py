"""
Simple MLP Agent for Permuted MNIST
Compatible with ML-Arena API.
Trainable, CPU-only, robust to input shapes (N,28,28), (N,784) or (784,N).
"""

import numpy as np
import torch
from torch import nn
import time


class Agent:
    def __init__(
        self,
        input_dim: int = 28 * 28,
        output_dim: int = 10,
        hidden=(1024, 512),
        dropout: float = 0.10,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        max_epochs: int = 10,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.1,
        label_smoothing: float = 0.05,
        time_budget_s: float = 60.0,
        seed: int = 42,
    ):
        # Fix random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cpu")

        # Store params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.label_smoothing = label_smoothing
        self.time_budget_s = time_budget_s

        # Build model & loss
        self.model = self._build_mlp().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # -------------------------------------------------------------------------
    # Architecture simple : Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear
    def _build_mlp(self):
        layers = []
        d = self.input_dim
        for h in self.hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, self.output_dim))
        return nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    def reset(self):
        """Réinitialise complètement le réseau (poids neufs)."""
        self.model = self._build_mlp().to(self.device)

    # -------------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne le MLP sous contrainte de temps."""

        # Correction d’orientation des données
        if X_train.shape[0] == self.input_dim and X_train.shape[1] != self.input_dim:
            X_train = X_train.T
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        X = torch.from_numpy(X_train).float().div_(255.0).to(self.device)
        y = torch.from_numpy(y_train).long().to(self.device)
        if y.ndim > 1:
            y = y.squeeze()

        # Split train/val
        N = X.size(0)
        idx = torch.randperm(N)
        n_val = int(self.val_fraction * N)
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        X_tr, y_tr = X[idx[n_val:]], y[idx[n_val:]]

        # Optimizer
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc, best_state = 0.0, None
        start = time.time()

        for ep in range(self.max_epochs):
            if time.time() - start > self.time_budget_s:
                break

            self.model.train()
            perm = torch.randperm(X_tr.size(0))
            for i in range(0, X_tr.size(0), self.batch_size):
                if time.time() - start > self.time_budget_s:
                    break
                b = perm[i:i+self.batch_size]
                xb, yb = X_tr[b], y_tr[b]
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                opt.step()

            # Validation rapide
            self.model.eval()
            with torch.no_grad():
                acc = (self.model(X_val).argmax(1) == y_val).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # -------------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prédit les classes sur X_test."""
        if X_test.shape[0] == self.input_dim and X_test.shape[1] != self.input_dim:
            X_test = X_test.T
        if X_test.ndim == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        X = torch.from_numpy(X_test).float().div_(255.0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).argmax(dim=1).cpu().numpy()
        return preds