# src/train_fixed.py
"""
Runnable training script for the credit default dataset.
Fixes applied:
 - Notebook-friendly ROOT (no __file__ usage)
 - Added missing RANDOM_STATE and TARGET constants
 - Robust CSV loading with helpful error messages
 - Defensive checks for required columns
 - Saves the pipeline artifact with model + metadata
 - Minor safety checks and explanatory prints

Usage (from notebook):
    %run src/train_fixed.py
or
    python src/train_fixed.py

Make sure your CSV is at: <project_root>/data/credit_data.csv or update DATA_PATH.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# ---------------- CONFIG ----------------
ROOT = os.path.abspath("")   # notebook-friendly
DATA_PATH = os.path.join(ROOT, "data", "credit_data.csv")
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RANDOM_STATE = 42
TARGET = "SeriousDlqin2yrs"   # 1 = default, 0 = no default

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------- Helpers -----------------

def load_data(path):
    """Load CSV into a DataFrame with a helpful error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find CSV at {path}. Put the file there or update DATA_PATH.")
    df = pd.read_csv(path)
    return df


def validate_columns(df):
    required = [
        TARGET,
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return required


def feature_engineering(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Cap utilization because extreme outliers may exist
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10)

    # Sum of delinquency counts (aggregate signal)
    df["num_delinquencies"] = (
        df.get("NumberOfTime30-59DaysPastDueNotWorse", 0).fillna(0)
        + df.get("NumberOfTime60-89DaysPastDueNotWorse", 0).fillna(0)
        + df.get("NumberOfTimes90DaysLate", 0).fillna(0)
    )

    # Derived flags
    if "DebtRatio" in df.columns:
        df["high_debt_ratio"] = (df["DebtRatio"] > 1).astype(int)

    # Income log (handle zeros / negatives and NaNs)
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome_clipped"] = df["MonthlyIncome"].clip(lower=0)
        df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome_clipped"])
        df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)

    if "NumberOfDependents" in df.columns:
        df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)

    # Interaction features
    df["util_x_open_accounts"] = (
        df.get("RevolvingUtilizationOfUnsecuredLines", 0).fillna(0)
        * df.get("NumberOfOpenCreditLinesAndLoans", 0).fillna(0)
    )

    # delinquencies per open account (avoid div by 0)
    num_open = df.get("NumberOfOpenCreditLinesAndLoans")
    if num_open is not None:
        df["delinq_per_account"] = df["num_delinquencies"] / (num_open.replace(0, np.nan).fillna(1))
    else:
        df["delinq_per_account"] = df["num_delinquencies"]

    return df


def build_pipeline(feature_cols):
    """Return a ColumnTransformer that imputes + scales numeric features."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, feature_cols)
    ], remainder="drop")

    return preprocessor


def evaluate_and_select(X_train, X_test, y_train, y_test, preprocessor):
    """Train candidate models and select the best by ROC AUC on the test set."""
    models = {}

    # Logistic Regression baseline
    pipe_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    pipe_lr.fit(X_train, y_train)
    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, proba_lr)
    models["logistic"] = {"pipeline": pipe_lr, "auc": auc_lr}

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    pipe_lgb = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", lgb)
    ])
    pipe_lgb.fit(X_train, y_train)
    proba_lgb = pipe_lgb.predict_proba(X_test)[:, 1]
    auc_lgb = roc_auc_score(y_test, proba_lgb)
    models["lightgbm"] = {"pipeline": pipe_lgb, "auc": auc_lgb}

    # Summary
    print("Model performance (ROC AUC):")
    for name, info in models.items():
        print(f" - {name}: {info['auc']:.4f}")

    # choose best by AUC
    best_name = max(models.keys(), key=lambda k: models[k]["auc"])
    best = models[best_name]
    print(f"Selected best model: {best_name} with AUC={best['auc']:.4f}")

    # classification report for chosen model
    best_pipe = best["pipeline"]
    preds = best_pipe.predict(X_test)
    print("Classification report for selected model (threshold 0.5):")
    print(classification_report(y_test, preds, digits=4))

    return best_name, best_pipe, models


# --------------- Train flow -----------------

def train():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    validate_columns(df)

    # Drop rows without target label
    df = df.dropna(subset=[TARGET]).copy()

    # Feature engineering
    df = feature_engineering(df)

    # Define final feature list (engineered + core)
    feature_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "DebtRatio",
        "MonthlyIncome",
        "MonthlyIncome_log",
        "MonthlyIncome_missing",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "NumberOfDependents_missing",
        "num_delinquencies",
        "high_debt_ratio",
        "util_x_open_accounts",
        "delinq_per_account"
    ]

    # Keep only columns that actually exist after feature engineering
    feature_cols = [c for c in feature_cols if c in df.columns]
    print("Using features:", feature_cols)

    X = df[feature_cols]
    y = df[TARGET].astype(int)

    # train-test split (stratified if possible)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    # build preprocessor
    preprocessor = build_pipeline(feature_cols)

    # evaluate models and choose best
    best_name, best_pipe, all_models = evaluate_and_select(X_train, X_test, y_train, y_test, preprocessor)

    # Save artifact: pipeline + meta
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "pipeline": best_pipe,
        "features": feature_cols,
        "target": TARGET,
        "model_name": best_name,
        "metrics": {k: float(v["auc"]) for k, v in all_models.items()}
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved best model artifact to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
