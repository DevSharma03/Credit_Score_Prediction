# src/app.py
import os
import logging
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer

# ---------------- CONFIG ----------------
_DATA_PATH_REL = "../data/credit_data.csv"
_MODEL_DIR_REL = "../model"
_MODEL_PATH_REL = "../model/model.pkl"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
TARGET_DEFAULT = "SeriousDlqin2yrs"

# Resolve relative paths reliably relative to this file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Now BASE_DIR points to project root (where model/ and data/ exist)
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-api")

# Load artifact
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run training to create the artifact."
    )

artifact = joblib.load(MODEL_PATH)

# Expected artifact keys:
# - "pipeline": sklearn pipeline or estimator with predict_proba
# - "features": list of feature names (order matters)
# - optionally "model_name", "target", "feature_medians"
pipeline = artifact.get("pipeline") or artifact.get("model") or None
FEATURES = artifact.get("features", None)
MODEL_NAME = artifact.get("model_name", "model")
TARGET = artifact.get("target", TARGET_DEFAULT)
FEATURE_MEDIANS = artifact.get("feature_medians", None)  # optional dict

if pipeline is None:
    raise RuntimeError("Loaded artifact does not contain a 'pipeline' or 'model' entry.")

if FEATURES is None:
    # try to infer features from artifact or dataset
    logger.warning("Artifact doesn't contain 'features' key. Attempting to infer from data CSV.")
    if os.path.exists(DATA_PATH):
        df_tmp = pd.read_csv(DATA_PATH)
        # drop target if present
        if TARGET in df_tmp.columns:
            df_tmp = df_tmp.drop(columns=[TARGET])
        FEATURES = [c for c in df_tmp.columns]
        # compute medians for fallback
        try:
            FEATURE_MEDIANS = df_tmp.median(numeric_only=True).to_dict()
        except Exception:
            FEATURE_MEDIANS = None
        logger.info("Inferred FEATURES from CSV: %s", FEATURES)
    else:
        raise RuntimeError("No FEATURES in artifact and no dataset available to infer them.")

# If no saved medians, attempt to compute from CSV dataset (best-effort)
if FEATURE_MEDIANS is None and os.path.exists(DATA_PATH):
    try:
        df_train = pd.read_csv(DATA_PATH)
        if TARGET in df_train.columns:
            df_train = df_train.drop(columns=[TARGET])
        FEATURE_MEDIANS = df_train.median(numeric_only=True).to_dict()
        logger.info("Computed feature medians from dataset for imputation.")
    except Exception as e:
        logger.warning("Could not compute medians from CSV (%s). Falling back to zeros.", e)
        FEATURE_MEDIANS = None

# Final fallback: use zeros for any missing median
if FEATURE_MEDIANS is None:
    FEATURE_MEDIANS = {f: 0.0 for f in FEATURES}
    logger.info("Using zero as fallback impute value for all features.")

# FastAPI app and schemas
app = FastAPI(title="Credit Risk Scoring API", version="1.0")

class ScoreRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Mapping of feature name -> value")

class TopFeature(BaseModel):
    feature: str
    importance: float

class ScoreResponse(BaseModel):
    probability_default: float
    prediction: int
    model: str
    top_features: List[TopFeature] = []

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "features_count": len(FEATURES)}

def _prepare_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Create ordered DataFrame with FEATURES and perform safe imputation:
    - convert pd.NA/None -> np.nan
    - fill missing columns using FEATURE_MEDIANS
    - ensure dtype cast to float for numeric columns
    """
    # Build single-row DataFrame
    df = pd.DataFrame([payload])

    # Add missing feature columns (explicitly) with pd.NA
    for c in FEATURES:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep only feature columns in expected order
    df = df[FEATURES]

    # Replace pandas NA / None with numpy NaN
    df = df.replace({pd.NA: np.nan, None: np.nan, "": np.nan})

    # For any non-numeric-like strings that may be present, try to coerce numeric columns
    # We'll attempt to convert all columns to numeric where possible; non-convertible will become NaN
    for col in df.columns:
        # If already numeric dtype, skip
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # coerce to numeric (errors -> NaN)
        coerced = pd.to_numeric(df[col], errors="coerce")
        df[col] = coerced

    # Impute missing values using FEATURE_MEDIANS (dictionary)
    # For columns missing in FEATURE_MEDIANS, fallback to column median from df or 0.0
    impute_values = {}
    col_medians = df.median(numeric_only=True)
    for col in df.columns:
        if col in FEATURE_MEDIANS and FEATURE_MEDIANS.get(col) is not None:
            impute_values[col] = FEATURE_MEDIANS[col]
        else:
            # use column median from input (if defined), else 0
            median_val = col_medians.get(col)
            impute_values[col] = float(median_val) if not pd.isna(median_val) else 0.0

    # Fill NaNs with impute_values
    df = df.fillna(value=impute_values)

    # As a safety, ensure all columns are numeric floats
    try:
        df = df.astype(float)
    except Exception as e:
        # If cast fails, raise an informative error
        raise ValueError(f"Failed to cast input features to float. Detail: {e}")

    return df

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    payload = req.data
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="'data' must be a JSON object of features.")

    # Prepare DataFrame and impute missing values
    try:
        df = _prepare_input_df(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during input preparation")
        raise HTTPException(status_code=500, detail=f"Input preparation error: {e}")

    # Predict probability
    try:
        # If pipeline expects a DataFrame with same column order, we already ensured FEATURES ordering.
        proba_arr = pipeline.predict_proba(df)
        # ensure shape safety
        if proba_arr.ndim == 1:
            # Some estimators might return only positive class, handle defensively
            proba = float(proba_arr[0])
        else:
            proba = float(proba_arr[:, 1][0])
        prediction = int(proba >= DEFAULT_THRESHOLD)
    except AttributeError:
        # pipeline doesn't support predict_proba; try score/predict fallback
        try:
            pred = pipeline.predict(df)[0]
            # If direct predict returns class label, set probability to 1.0 for predicted class (best-effort)
            proba = float(pred)
            prediction = int(pred)
        except Exception as e:
            logger.exception("Model prediction failed (no predict_proba and predict fallback failed)")
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    except Exception as e:
        logger.exception("Model prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Try to compute top feature importances if underlying estimator has them
    top_features_list: List[Dict[str, Any]] = []
    try:
        # Best-effort: get estimator from named_steps['clf'] or last pipeline step
        est = getattr(pipeline, "named_steps", {}).get("clf", None)
        if est is None and hasattr(pipeline, "steps"):
            est = pipeline.steps[-1][1]
        if est is not None and hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
            # ensure length matches FEATURES
            if len(importances) == len(FEATURES):
                feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)[:5]
                top_features_list = [{"feature": k, "importance": float(v)} for k, v in feat_imp]
    except Exception:
        # silently ignore importance extraction errors
        top_features_list = []

    return ScoreResponse(
        probability_default=proba,
        prediction=prediction,
        model=MODEL_NAME,
        top_features=top_features_list,
    )

# Optional: simple metadata endpoint to help debugging
@app.get("/metadata")
def metadata():
    return {
        "model": MODEL_NAME,
        "num_features": len(FEATURES),
        "features_preview": FEATURES[:50],
        "impute_values_provided": bool(artifact.get("feature_medians", None)),
        "data_path_exists": os.path.exists(DATA_PATH),
    }

