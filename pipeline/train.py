"""
Training pipeline for 3D soil profile prediction.

Stage 1 : Classify soil_layer from (easting, northing, depth_m)
Stage 2 : Four regressors — su_kpa (ST rows), spt_n (SS rows),
          unit_weight (all rows), plasticity_idx (rows where available)

Methods  : distance-weighted average (baseline), Random Forest, XGBoost
Validation: leave-one-out cross-validation by borehole
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

from preprocess import load_and_clean, engineer_features, COHESIVE_LAYERS, GRANULAR_LAYERS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bangkok_boring_logs_real.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Baseline: distance-weighted average
# ---------------------------------------------------------------------------

def _haversine_2d(e1, n1, e2, n2):
    return np.sqrt((e1 - e2) ** 2 + (n1 - n2) ** 2)


def dwa_predict_class(train_df, query_e, query_n, query_d, k=5):
    """Distance-weighted majority vote for soil_layer."""
    if train_df.empty:
        return "VSC"  # fallback when no training data (e.g. single-borehole LOO)
    dists = np.sqrt(
        (train_df["easting"] - query_e) ** 2
        + (train_df["northing"] - query_n) ** 2
        + (train_df["depth_m"] - query_d) ** 2
    )
    dists = dists.replace(0, 1e-6)
    top = train_df.assign(_d=dists).nsmallest(k, "_d")
    weights = 1.0 / top["_d"].values
    # accumulate weighted votes per layer
    vote = {}
    for lyr, w in zip(top["soil_layer"].values, weights):
        vote[lyr] = vote.get(lyr, 0.0) + w
    return max(vote, key=vote.__getitem__)


def dwa_predict_value(train_df, col, query_e, query_n, query_d, k=5):
    """Distance-weighted average for a numeric target column."""
    sub = train_df.dropna(subset=[col])
    if sub.empty:
        return np.nan
    dists = np.sqrt(
        (sub["easting"] - query_e) ** 2
        + (sub["northing"] - query_n) ** 2
        + (sub["depth_m"] - query_d) ** 2
    )
    dists = dists.replace(0, 1e-6)
    top = sub.assign(_d=dists).nsmallest(k, "_d")
    weights = 1.0 / top["_d"]
    return float((top[col] * weights.values).sum() / weights.sum())


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

ALL_LAYERS = ["MG", "SOC", "MSC", "SC", "FS", "VSC", "SS"]


def make_stage1_X(df):
    return df[["easting", "northing", "depth_m"]].values


def make_stage2_X(df):
    """easting, northing, depth_m + one-hot soil_layer (fixed column order)."""
    base = df[["easting", "northing", "depth_m"]].copy()
    for layer in ALL_LAYERS:
        base[f"layer_{layer}"] = (df["soil_layer"] == layer).astype(int)
    return base.values


# ---------------------------------------------------------------------------
# Leave-one-out CV by borehole
# ---------------------------------------------------------------------------

def loo_cv_classifier(df, model_fn, predict_fn):
    boreholes = df["borehole_id"].unique()
    if len(boreholes) < 2:
        print("  WARNING: only 1 borehole — LOO-CV will use self-prediction (optimistic)")
    y_true, y_pred = [], []
    for bh in boreholes:
        train = df[df["borehole_id"] != bh]
        test = df[df["borehole_id"] == bh]
        # fall back to full dataset when train is empty (single borehole)
        train_for_model = train if not train.empty else df
        model = model_fn(train_for_model)
        preds = [predict_fn(model, train_for_model, row) for _, row in test.iterrows()]
        y_true.extend(test["soil_layer"].tolist())
        y_pred.extend(preds)
    return y_true, y_pred


def loo_cv_regressor(df, col, model_fn, predict_fn):
    sub = df.dropna(subset=[col])
    boreholes = sub["borehole_id"].unique()
    y_true, y_pred = [], []
    for bh in boreholes:
        train = sub[sub["borehole_id"] != bh]
        test = sub[sub["borehole_id"] == bh]
        if train.empty or test.empty:
            continue
        model = model_fn(train)
        preds = [predict_fn(model, train, row, col) for _, row in test.iterrows()]
        y_true.extend(test[col].tolist())
        y_pred.extend(preds)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Model factories & predictors
# ---------------------------------------------------------------------------

label_enc = LabelEncoder().fit(ALL_LAYERS)


def _rf_classifier(train):
    X = make_stage1_X(train)
    y = label_enc.transform(train["soil_layer"])
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf


def _xgb_classifier(train):
    X = make_stage1_X(train)
    # Use a per-fold encoder so labels are always [0..n_classes-1]
    # (global label_enc can produce gaps when a class is absent from the fold)
    fold_enc = LabelEncoder().fit(train["soil_layer"])
    y = fold_enc.transform(train["soil_layer"])
    clf = XGBClassifier(n_estimators=200, random_state=42,
                        eval_metric="mlogloss", verbosity=0)
    clf.fit(X, y)
    clf._fold_enc = fold_enc  # carry encoder alongside model for inverse-transform
    return clf


def _rf_clf_predict(model, train, row):
    x = np.array([[row["easting"], row["northing"], row["depth_m"]]])
    return label_enc.inverse_transform(model.predict(x))[0]


def _xgb_clf_predict(model, train, row):
    x = np.array([[row["easting"], row["northing"], row["depth_m"]]])
    fold_idx = int(model.predict(x)[0])
    return model._fold_enc.inverse_transform([fold_idx])[0]


def _dwa_clf_predict(_, train, row):
    return dwa_predict_class(train, row["easting"], row["northing"], row["depth_m"])


def _rf_regressor(train):
    X = make_stage2_X(train)
    return X  # placeholder — actual model built per-target below


def _make_rf_reg(train, col):
    sub = train.dropna(subset=[col])
    X = make_stage2_X(sub)
    y = sub[col].values
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X, y)
    return reg


def _make_xgb_reg(train, col):
    sub = train.dropna(subset=[col])
    X = make_stage2_X(sub)
    y = sub[col].values
    reg = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    reg.fit(X, y)
    return reg


def _reg_predict_row(model, row):
    base = [row["easting"], row["northing"], row["depth_m"]]
    layer_oh = [int(row["soil_layer"] == lyr) for lyr in ALL_LAYERS]
    x = np.array([base + layer_oh])
    return float(model.predict(x)[0])


# Regressor LOO wrappers that build a fresh model per fold
def _loo_reg(df, col, build_fn):
    sub = df.dropna(subset=[col])
    boreholes = sub["borehole_id"].unique()
    single_bh = len(boreholes) < 2
    y_true, y_pred = [], []
    for bh in boreholes:
        train_fold = sub[sub["borehole_id"] != bh]
        test_fold = sub[sub["borehole_id"] == bh]
        # single-borehole fallback: train on full set (self-prediction)
        if len(train_fold) < 2:
            train_fold = sub
        model = build_fn(train_fold, col)
        for _, row in test_fold.iterrows():
            y_true.append(row[col])
            y_pred.append(_reg_predict_row(model, row))
    return y_true, y_pred


def _loo_dwa_reg(df, col):
    sub = df.dropna(subset=[col])
    boreholes = sub["borehole_id"].unique()
    y_true, y_pred = [], []
    for bh in boreholes:
        train_fold = sub[sub["borehole_id"] != bh]
        test_fold = sub[sub["borehole_id"] == bh]
        # single-borehole fallback: use adjacent rows within same borehole
        if train_fold.empty:
            train_fold = sub
        for _, row in test_fold.iterrows():
            y_true.append(row[col])
            y_pred.append(
                dwa_predict_value(train_fold, col, row["easting"], row["northing"], row["depth_m"])
            )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def clf_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  [{name}] accuracy={acc:.3f}  weighted-F1={f1:.3f}")
    return {"accuracy": acc, "f1_weighted": f1}


def reg_metrics(y_true, y_pred, name, col):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 2:
        print(f"  [{name}] {col}: insufficient data")
        return {}
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    r2 = r2_score(y_true[mask], y_pred[mask])
    print(f"  [{name}] {col}: MAE={mae:.3f}  R²={r2:.3f}  (n={mask.sum()})")
    return {"mae": mae, "r2": r2, "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(data_path=DATA_PATH, model_dir=MODEL_DIR):
    print("Loading data …")
    df = load_and_clean(data_path)
    print(f"  {len(df)} rows, {df['borehole_id'].nunique()} boreholes")

    results = {}

    # -----------------------------------------------------------------------
    # Stage 1 — soil layer classification (LOO CV)
    # -----------------------------------------------------------------------
    print("\n=== Stage 1: Soil Layer Classification (LOO-CV) ===")
    results["stage1"] = {}

    # DWA baseline
    yt, yp = loo_cv_classifier(df, lambda t: None, _dwa_clf_predict)
    results["stage1"]["dwa"] = clf_metrics(yt, yp, "DWA")

    # Random Forest
    yt, yp = loo_cv_classifier(df, _rf_classifier, _rf_clf_predict)
    results["stage1"]["rf"] = clf_metrics(yt, yp, "RF")

    # XGBoost
    yt, yp = loo_cv_classifier(df, _xgb_classifier, _xgb_clf_predict)
    results["stage1"]["xgb"] = clf_metrics(yt, yp, "XGB")

    # Train final Stage 1 models on full data and save
    print("\nTraining final Stage 1 models on full dataset …")
    rf_clf = _rf_classifier(df)
    xgb_clf = _xgb_classifier(df)
    joblib.dump(rf_clf, os.path.join(model_dir, "stage1_rf_classifier.joblib"))
    joblib.dump(xgb_clf, os.path.join(model_dir, "stage1_xgb_classifier.joblib"))
    joblib.dump(label_enc, os.path.join(model_dir, "label_encoder.joblib"))

    # -----------------------------------------------------------------------
    # Stage 2 — property regression (LOO CV)
    # -----------------------------------------------------------------------
    # su_kpa  : rows where su_method in {ST}   (Shelby Tube samples only)
    # spt_n   : rows where su_method is blank  (SS / Split Spoon samples)
    # unit_weight : all rows with valid values
    # plasticity_idx : all rows with valid values (not restricted to clay)

    su_df = df[df["su_method"] == "ST"].copy()
    ss_df = df[df["su_method"].isna() | (df["su_method"] == "")].copy()

    reg_targets = {
        "su_kpa": su_df,
        "spt_n": ss_df,
        "unit_weight": df,
        "plasticity_idx": df,
    }

    print("\n=== Stage 2: Property Regression (LOO-CV) ===")
    results["stage2"] = {}

    for col, source_df in reg_targets.items():
        print(f"\n--- {col} ---")
        results["stage2"][col] = {}

        yt, yp = _loo_dwa_reg(source_df, col)
        results["stage2"][col]["dwa"] = reg_metrics(yt, yp, "DWA", col)

        yt, yp = _loo_reg(source_df, col, _make_rf_reg)
        results["stage2"][col]["rf"] = reg_metrics(yt, yp, "RF", col)

        yt, yp = _loo_reg(source_df, col, _make_xgb_reg)
        results["stage2"][col]["xgb"] = reg_metrics(yt, yp, "XGB", col)

    # Train final Stage 2 models on full data and save
    print("\nTraining final Stage 2 models on full dataset …")
    for col, source_df in reg_targets.items():
        sub = source_df.dropna(subset=[col])
        if len(sub) < 2:
            print(f"  Skipping {col} — not enough rows ({len(sub)})")
            continue

        rf_reg = _make_rf_reg(sub, col)
        xgb_reg = _make_xgb_reg(sub, col)
        joblib.dump(rf_reg, os.path.join(model_dir, f"stage2_rf_{col}.joblib"))
        joblib.dump(xgb_reg, os.path.join(model_dir, f"stage2_xgb_{col}.joblib"))
        print(f"  Saved models for {col}")

    # Persist CV results
    results_path = os.path.join(model_dir, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCV results saved to {results_path}")

    return results


if __name__ == "__main__":
    train()
