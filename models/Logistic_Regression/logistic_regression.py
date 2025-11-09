import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from sklearn.linear_model import LogisticRegression

class Agent:
    def __init__(self, C: float = 0.1, max_iter: int = 100, n_jobs: int = 2, seed: int = 42):
        """Initialize your agent"""
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.seed = seed

        self.reset()
        return None

    def reset(self):
        """Reset for a new task (new permutation)"""
        # Réinitialise le modèle pour une nouvelle tâche
        self.model = LogisticRegression(
            penalty='l2',               # régularisation standard
            C=self.C,                   # force de régularisation
            solver='lbfgs',             # bon compromis pour données denses
            max_iter=self.max_iter,     # nécessaire pour convergence
            n_jobs=self.n_jobs,         # parallelisation
            random_state=self.seed      # reproductibilité
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train on the permuted training data"""
        # Flatten images if needed (28x28 -> 784)
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)

        # Normalize to [0, 1]
        if X_train.max() > 1:
            X_train = X_train.astype(np.float32) / 255.0

        # Flatten labels if needed
        y_train = y_train.ravel()
        
        # Fit the logistic regression model
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions for test data"""
        if X_test.ndim > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        if X_test.max() > 1:
            X_test = X_test.astype(np.float32) / 255.0

        return self.model.predict(X_test)