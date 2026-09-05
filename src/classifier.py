from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


def get_default_model(name: str):
    """Factory function to get a fresh instance of a classifier by name."""
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
    elif name == "Neural Network (MLP)":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


class ProductionClassifier:
    """Production classifier engine supporting Random Forest, Gradient Boosting, and Neural Network (MLP) with dynamic switching."""

    def __init__(self) -> None:
        self.models = {
            "Random Forest": get_default_model("Random Forest"),
            "Gradient Boosting": get_default_model("Gradient Boosting"),
            "Neural Network (MLP)": get_default_model("Neural Network (MLP)"),
        }
        self.active_model_name = "Random Forest"
        self._is_fitted = {name: False for name in self.models}

    def set_active_model(self, model_name: str) -> None:
        """Switch the active classifier model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}. Choose from {list(self.models.keys())}")
        self.active_model_name = model_name

    def fit(self, X: np.ndarray, y: np.ndarray, model_name: str | None = None) -> None:
        """Train classifier models on features and labels. Handles 1-class datasets using a DummyClassifier fallback."""
        unique_classes = np.unique(y)
        has_multiple_classes = len(unique_classes) >= 2
        
        models_to_fit = [model_name] if model_name is not None else list(self.models.keys())
        
        for name in models_to_fit:
            if has_multiple_classes:
                # If we now have both classes, ensure we are using the real model (not dummy fallback)
                if isinstance(self.models[name], DummyClassifier):
                    print(f"[Classifier] Dynamic upgrade: re-instantiating real '{name}' model for 2-class training.")
                    self.models[name] = get_default_model(name)
                print(f"[Classifier] Fitting real model: {name}...")
                self.models[name].fit(X, y)
                self._is_fitted[name] = True
            else:
                # 1 class present
                if name == "Random Forest":
                    # Random Forest in scikit-learn supports 1-class training natively
                    print(f"[Classifier] Fitting model: {name} (native 1-class support)...")
                    self.models[name].fit(X, y)
                    self._is_fitted[name] = True
                else:
                    print(f"[Classifier] Only 1 class present. Using DummyClassifier fallback for: {name}")
                    dummy = DummyClassifier(strategy="most_frequent")
                    dummy.fit(X, y)
                    self.models[name] = dummy
                    self._is_fitted[name] = True

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Return (class prediction, max class probability) for one sample using the active model."""
        if not self._is_fitted[self.active_model_name]:
            raise RuntimeError(f"Active classifier '{self.active_model_name}' must be fit before predict.")
        
        model = self.models[self.active_model_name]
        X = features.reshape(1, -1)
        pred = float(model.predict(X)[0])
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0
            
        return pred, confidence

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for a batch of raw feature rows using the active model."""
        if not self._is_fitted[self.active_model_name]:
            raise RuntimeError(f"Active classifier '{self.active_model_name}' must be fit before predict_batch.")
        return self.models[self.active_model_name].predict(X).astype(np.int32)
