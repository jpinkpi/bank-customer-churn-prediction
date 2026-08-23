"""Train and evaluate models for bank customer churn."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Exited"
IDENTIFIERS = ("RowNumber", "CustomerId", "Surname")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the churn CSV file")
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}'.")
    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET, *IDENTIFIERS], errors="ignore")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns
    categorical = X.columns.difference(numeric)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, probabilities >= value) for value in candidates]
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def main() -> None:
    args = parse_args()
    X, y = load_data(args.data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    estimators = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(class_weight="balanced", n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }

    candidates = []
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        valid_probability = pipeline.predict_proba(X_valid)[:, 1]
        threshold, validation_f1 = best_threshold(y_valid, valid_probability)
        candidates.append((validation_f1, name, threshold, pipeline))
        print(f"{name}: validation F1={validation_f1:.3f}, threshold={threshold:.2f}")

    _, name, threshold, model = max(candidates, key=lambda item: item[0])
    test_probability = model.predict_proba(X_test)[:, 1]
    prediction = (test_probability >= threshold).astype(int)
    print(f"\nSelected model: {name}")
    print(f"Test F1:       {f1_score(y_test, prediction):.3f}")
    print(f"Test precision:{precision_score(y_test, prediction):.3f}")
    print(f"Test recall:   {recall_score(y_test, prediction):.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, prediction):.3f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, test_probability):.3f}")


if __name__ == "__main__":
    main()
