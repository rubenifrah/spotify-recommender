from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from typing import Union, Any

def train_xgboost(X_train: np.ndarray, y_train: pd.Series) -> XGBClassifier:
    """
    Trains an XGBoost classifier with pre-tuned hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        XGBClassifier: Trained model.
    """
    print("Training XGBoost Classifier...")
    # Parameters taken from your notebook's GridSearch results
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train: np.ndarray, y_train: pd.Series) -> MLPClassifier:
    """
    Trains an MLP Classifier (Neural Network).

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        MLPClassifier: Trained model.
    """
    print("Training MLP Classifier (Neural Network)...")
    # Parameters from your notebook (approx best params)
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: np.ndarray, y_test: pd.Series) -> None:
    """
    Evaluates the model on the test set and prints classification report and confusion matrix.

    Args:
        model: Trained model (XGBoost or MLP).
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))