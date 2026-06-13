"""wrappers.py — small, clone-safe sklearn estimators.

CatBoostWrapper exists for one reason: sklearn's clone() rejects a bare
CatBoostClassifier that carries `cat_features` in its constructor ("the
constructor either does not set or modifies parameter cat_features"), because
CatBoost rewrites that attribute during fit. Anything that clones the estimator —
CalibratedClassifierCV, cross_val_predict, GridSearchCV — therefore blows up.

This thin wrapper stores every hyperparameter as a plain __init__ attribute and
never mutates it (the real CatBoostClassifier is built fresh inside fit and kept
as self.model_). get_params() round-trips cleanly, so clone() works — which lets
CalibratedClassifierCV cross-validate the calibration without leakage.

It is deliberately importable (not defined in a __main__ script) so a model
pickled with it — e.g. the registered calibrated model — can be loaded by the
serving process, which imports the same class.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """sklearn-compatible CatBoostClassifier with categoricals passed by name.

    Every constructor argument is explicit (no *args/**kwargs) so sklearn's
    get_params/clone introspection works. `cat_features` is a list of column
    names; the input X must be a DataFrame whose categorical columns are strings.
    """

    def __init__(
        self,
        cat_features=None,
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        border_count=254,
        auto_class_weights="Balanced",
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
    ):
        self.cat_features = cat_features
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.auto_class_weights = auto_class_weights
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.model_ = CatBoostClassifier(
            cat_features=self.cat_features,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            border_count=self.border_count,
            auto_class_weights=self.auto_class_weights,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).ravel()
