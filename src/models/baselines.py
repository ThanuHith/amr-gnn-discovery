"""Random Forest and XGBoost multi-task baseline models for AMR discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


class MultiTaskAMRBaseline:
    """Multi-task classical ML baseline trained on 2048-bit Morgan FP + 12 eNTRy descriptors."""

    def __init__(self, model_type: str = "xgb", random_state: int = 42) -> None:
        if model_type not in {"rf", "xgb"}:
            raise ValueError("model_type must be 'rf' or 'xgb'")
        self.model_type = model_type
        self.random_state = random_state
        self.models: Dict[str, object] = {}

    def _make_clf(self) -> object:
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=260,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state,
            )
        return XGBClassifier(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.5,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _make_reg(self) -> object:
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=260,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
            )
        return XGBRegressor(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.5,
            eval_metric="rmse",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_dict: Dict[str, np.ndarray],
    ) -> "MultiTaskAMRBaseline":
        """Fit all 5 multi-task estimators (3 classifiers + 2 regressors)."""
        task_specs = {
            "active_abx": "clf",
            "pMIC": "reg",
            "gram_neg_accum": "clf",
            "cytotoxic_human": "clf",
            "pCC50": "reg",
        }
        for task, kind in task_specs.items():
            est = self._make_clf() if kind == "clf" else self._make_reg()
            est.fit(X_train, y_dict[task])
            self.models[task] = est
        return self

    def predict_dict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict probabilities and continuous potency/toxicity values."""
        return {
            "prob_abx": self.models["active_abx"].predict_proba(X)[:, 1],
            "pred_pMIC": self.models["pMIC"].predict(X),
            "prob_gram_neg": self.models["gram_neg_accum"].predict_proba(X)[:, 1],
            "prob_cytotox": self.models["cytotoxic_human"].predict_proba(X)[:, 1],
            "pred_pCC50": self.models["pCC50"].predict(X),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "MultiTaskAMRBaseline":
        return joblib.load(path)
