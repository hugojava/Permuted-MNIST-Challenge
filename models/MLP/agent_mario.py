"""
name_file = agent_Bruce_Wayne.py
Permuted MNIST — MLP Submission (fast, BN-ready)

Chosen config (from your Optuna summary, Phase C 2L TOP):
hidden=(2048, 1024), dropout=0.05, batch_size=1024,
learning_rate=0.0015, weight_decay=1e-4,
label_smoothing=0.05, max_epochs=10, val_fraction=0.10,
batch_norm=True, patience=4, eval_every=2. CPU-only.

Observed locally in your runs:
acc ≈ 0.9867 ± 0.0003 | t ≈ 36.9 s (10 tasks)
"""

import os, time, numpy as np, torch
from torch import nn

# --- CPU-only constraints (before torch/numpy) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def build_mlp(
    input_dim: int = 784,
    hidden: tuple[int, ...] = (2048, 1024),
    dropout: float = 0.05,
    num_classes: int = 10,
    batch_norm: bool = True,
):
    layers: list[nn.Module] = []
    d = input_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, num_classes))
    net = nn.Sequential(*layers)

    # Kaiming init for linear layers
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


class Agent:
    def __init__(self, output_dim: int = 10, seed: int | None = None):
        # ---- Optuna best (2L with BN) ----
        self.input_dim = 28 * 28
        self.num_classes = int(output_dim)
        self.hidden = (2048, 1024)
        self.dropout = 0.05
        self.batch_norm = True

        self.batch_size = 1024
        self.learning_rate = 1.5e-3
        self.weight_decay = 1e-4
        self.label_smoothing = 0.05

        self.max_epochs = 10
        self.val_fraction = 0.10
        self.patience = 4
        self.eval_every = 2

        # Safety guard for platform limits
        self.time_budget_s = 58.0

        if seed is None:
            seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.model = build_mlp(
            input_dim=self.input_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            num_classes=self.num_classes,
            batch_norm=self.batch_norm,
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def reset(self):
        self.model = build_mlp(
            input_dim=self.input_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            num_classes=self.num_classes,
            batch_norm=self.batch_norm,
        ).to(self.device)

    # ------------ helpers (robust to (N,28,28), (N,784), or (784,N)) ------------
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        # If features first: (784, N) -> (N, 784)
        if X.ndim == 2 and X.shape[0] == self.input_dim and X.shape[1] != self.input_dim:
            X = X.T
        # If images: (N, 28, 28) -> (N, 784)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = torch.from_numpy(X).float()
        if X.numel() > 0 and X.max() > 1.0 + 1e-6:  # uint8 [0..255]
            X = X / 255.0
        return X

    def _to_labels(self, y: np.ndarray) -> torch.Tensor:
        if y.ndim > 1:
            y = y.squeeze()
        return torch.from_numpy(y).long()

    # -------------------------------- train/predict --------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X = self._to_tensor(X_train).to(self.device)
        y = self._to_labels(y_train).to(self.device)

        N = X.size(0)
        idx = torch.randperm(N)
        n_val = int(self.val_fraction * N)
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        X_tr,  y_tr  = X[idx[n_val:]], y[idx[n_val:]]

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_acc, best_state = 0.0, None
        patience = self.patience
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
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()

            # light validation (every eval_every epochs or last one)
            if ((ep + 1) % self.eval_every == 0) or (ep + 1 == self.max_epochs):
                self.model.eval()
                with torch.no_grad():
                    acc = (self.model(X_val).argmax(1) == y_val).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    patience = self.patience
                else:
                    patience -= 1
                    if patience == 0:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = self._to_tensor(X_test).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1).cpu().numpy()