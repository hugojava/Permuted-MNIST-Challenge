import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from xgboost import XGBClassifier

class Agent:
    def __init__(self, n_estimators: int = 25, max_depth: int = 5, learning_rate: float = 0.66, n_jobs: int = 2, seed: int = 42):
        """Initialize your agent"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample=0.8
        self.colsample_bytree=0.8
        self.n_jobs = n_jobs
        self.seed = seed

        self.reset()
        return None

    def reset(self):
        """Reset for a new task (new permutation)"""
        # Réinitialise le modèle pour une nouvelle tâche
        self.model = XGBClassifier(
            n_estimators = self.n_estimators,       # number of estimators
            max_depth = self.max_depth,             # maximun depth of each estimator
            learning_rate = self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs = self.n_jobs,                   # parallelisation
            random_state = self.seed
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