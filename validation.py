"""
Leave-One-Out Cross-Validation by Borehole
===========================================
Evaluates three soil prediction methods on the Bangkok boring-log dataset:
  1. Distance-Weighted Average (DWA)  — no model files needed
  2. Random Forest (RF)               — requires trained .joblib files
  3. XGBoost (XGB)                    — requires trained .joblib files

Metrics reported
  - Soil Layer Classification : Accuracy (%)
  - Unit Weight               : RMSE, MAE
  - Su (kPa)                  : RMSE, MAE
  - SPT-N                     : RMSE, MAE

Usage
-----
  python validation.py                          # uses default data / model paths
  python validation.py --data my_data.csv       # custom CSV
  python validation.py --models my_models/      # custom model directory
  python validation.py --out results.json       # save results to JSON

The script saves results to  validation_results.json  by default.
Upload that file in the app's "Model Validation" tab to display the results.
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script so it can run from any directory
# ---------------------------------------------------------------------------
_ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_ROOT, "data", "bangkok_boring_logs_real.csv")
MODEL_DIR  = os.path.join(_ROOT, "models")

# ---------------------------------------------------------------------------
# Soil layer constants (must match train.py)
# ---------------------------------------------------------------------------
ALL_LAYERS = ["MG", "SOC", "MSC", "SC", "FS", "VSC", "SS"]

# ---------------------------------------------------------------------------
# Data loading  (inline — no dependency on pipeline package)
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "depth_m", "depth_top_m", "depth_bot_m",
        "su_kpa", "spt_n", "unit_weight",
        "plasticity_idx", "liquid_limit", "plastic_limit", "water_content",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce mutual exclusivity: su_kpa only for ST/FV rows; spt_n only for SS rows
    if "su_method" in df.columns:
        has_su = df["su_method"].notna() & (df["su_method"] != "")
        df.loc[has_su, "spt_n"]   = np.nan
        df.loc[~has_su, "su_kpa"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Feature constructors
# ---------------------------------------------------------------------------

def _stage1_X(df: pd.DataFrame) -> np.ndarray:
    return df[["easting", "northing", "depth_m"]].values


def _stage2_X(df: pd.DataFrame) -> np.ndarray:
    base = df[["easting", "northing", "depth_m"]].copy()
    for lyr in ALL_LAYERS:
        base[f"layer_{lyr}"] = (df["soil_layer"] == lyr).astype(int)
    return base.values


# ---------------------------------------------------------------------------
# Distance-Weighted Average (baseline)
# ---------------------------------------------------------------------------

def _dwa_class(train: pd.DataFrame, e, n, d, k=5) -> str:
    if train.empty:
        return "VSC"
    dists = np.sqrt(
        (train["easting"] - e) ** 2
        + (train["northing"] - n) ** 2
        + (train["depth_m"] - d) ** 2
    ).replace(0, 1e-6)
    top = train.assign(_d=dists).nsmallest(k, "_d")
    vote: dict[str, float] = {}
    for lyr, w in zip(top["soil_layer"].values, 1.0 / top["_d"].values):
        vote[lyr] = vote.get(lyr, 0.0) + w
    return max(vote, key=vote.__getitem__)


def _dwa_value(train: pd.DataFrame, col: str, e, n, d, k=5) -> float:
    sub = train.dropna(subset=[col])
    if sub.empty:
        return np.nan
    dists = np.sqrt(
        (sub["easting"] - e) ** 2
        + (sub["northing"] - n) ** 2
        + (sub["depth_m"] - d) ** 2
    ).replace(0, 1e-6)
    top = sub.assign(_d=dists).nsmallest(k, "_d")
    w = 1.0 / top["_d"].values
    return float((top[col].values * w).sum() / w.sum())


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------

def _load_models(model_dir: str) -> dict:
    import joblib
    required = (
        ["label_encoder", "stage1_rf_classifier", "stage1_xgb_classifier"]
        + [f"stage2_rf_{t}"  for t in ["su_kpa", "spt_n", "unit_weight"]]
        + [f"stage2_xgb_{t}" for t in ["su_kpa", "spt_n", "unit_weight"]]
    )
    models = {}
    missing = []
    for name in required:
        p = os.path.join(model_dir, f"{name}.joblib")
        if not os.path.exists(p):
            missing.append(p)
        else:
            models[name] = joblib.load(p)
    if missing:
        print("\n  WARNING: Missing model files — RF and XGBoost skipped:")
        for p in missing:
            print(f"    {p}")
        print("  Run  python pipeline/train.py  to generate them.\n")
        return {}
    return models


def _rf_predict_class(model, enc, row) -> str:
    x = np.array([[row["easting"], row["northing"], row["depth_m"]]])
    return enc.inverse_transform(model.predict(x))[0]


def _xgb_predict_class(model, row) -> str:
    x = np.array([[row["easting"], row["northing"], row["depth_m"]]])
    idx = int(model.predict(x)[0])
    return model._fold_enc.inverse_transform([idx])[0]


def _reg_predict_row(model, row) -> float:
    base  = [row["easting"], row["northing"], row["depth_m"]]
    oh    = [int(row["soil_layer"] == lyr) for lyr in ALL_LAYERS]
    x     = np.array([base + oh])
    return float(max(0.0, model.predict(x)[0]))


# ---------------------------------------------------------------------------
# LOO-CV runners
# ---------------------------------------------------------------------------

def _loo_classify_dwa(df: pd.DataFrame):
    y_true, y_pred = [], []
    for bh in df["borehole_id"].unique():
        train = df[df["borehole_id"] != bh]
        test  = df[df["borehole_id"] == bh]
        if train.empty:
            train = df
        for _, row in test.iterrows():
            y_true.append(row["soil_layer"])
            y_pred.append(_dwa_class(train, row["easting"], row["northing"], row["depth_m"]))
    return y_true, y_pred


def _loo_classify_ml(df: pd.DataFrame, models: dict, method: str):
    """Re-train the classifier on each LOO fold and predict."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    global_enc = models["label_encoder"]
    y_true, y_pred = [], []

    for bh in df["borehole_id"].unique():
        train = df[df["borehole_id"] != bh]
        test  = df[df["borehole_id"] == bh]
        if len(train) < 2:
            train = df

        X_tr = _stage1_X(train)

        if method == "rf":
            y_tr = global_enc.transform(train["soil_layer"])
            clf  = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_tr, y_tr)
            enc  = global_enc
        else:  # xgb
            fold_enc = LabelEncoder().fit(train["soil_layer"])
            y_tr = fold_enc.transform(train["soil_layer"])
            clf = XGBClassifier(n_estimators=200, random_state=42,
                                eval_metric="mlogloss", verbosity=0)
            clf.fit(X_tr, y_tr)
            clf._fold_enc = fold_enc
            enc = None  # not used for xgb path

        for _, row in test.iterrows():
            y_true.append(row["soil_layer"])
            x = np.array([[row["easting"], row["northing"], row["depth_m"]]])
            if method == "rf":
                y_pred.append(enc.inverse_transform(clf.predict(x))[0])
            else:
                idx = int(clf.predict(x)[0])
                y_pred.append(clf._fold_enc.inverse_transform([idx])[0])

    return y_true, y_pred


def _loo_regress_dwa(df: pd.DataFrame, col: str):
    sub = df.dropna(subset=[col])
    y_true, y_pred = [], []
    for bh in sub["borehole_id"].unique():
        train = sub[sub["borehole_id"] != bh]
        test  = sub[sub["borehole_id"] == bh]
        if train.empty:
            train = sub
        for _, row in test.iterrows():
            y_true.append(row[col])
            y_pred.append(_dwa_value(train, col, row["easting"], row["northing"], row["depth_m"]))
    return y_true, y_pred


def _loo_regress_ml(df: pd.DataFrame, col: str, method: str):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    sub = df.dropna(subset=[col])
    y_true, y_pred = [], []
    for bh in sub["borehole_id"].unique():
        train = sub[sub["borehole_id"] != bh]
        test  = sub[sub["borehole_id"] == bh]
        if len(train) < 2:
            train = sub

        X_tr = _stage2_X(train)
        y_tr = train[col].values

        if method == "rf":
            reg = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            reg = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        reg.fit(X_tr, y_tr)

        for _, row in test.iterrows():
            y_true.append(row[col])
            y_pred.append(_reg_predict_row(reg, row))

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _accuracy(y_true, y_pred) -> float:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true) if y_true else 0.0


def _rmse_mae(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    if mask.sum() < 2:
        return np.nan, np.nan, 0
    err  = yt[mask] - yp[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    return rmse, mae, int(mask.sum())


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _hbar(char="─", width=74):
    print(char * width)


def _print_results(results: dict):
    _hbar("═")
    print("  LEAVE-ONE-OUT CROSS-VALIDATION BY BOREHOLE — RESULTS SUMMARY")
    _hbar("═")

    # Classification
    print()
    print("  SOIL LAYER CLASSIFICATION")
    _hbar()
    print(f"  {'Method':<30} {'Accuracy (%)':>14} {'N samples':>12}")
    _hbar()
    clsf = results["classification"]
    n_cls = clsf.get("n_samples", "?")
    for method in ["DWA", "RF", "XGB"]:
        key  = method.lower()
        acc  = clsf.get(key, {}).get("accuracy", np.nan)
        acc_s = f"{acc*100:.1f}" if not np.isnan(acc) else "n/a"
        print(f"  {method:<30} {acc_s:>14} {str(n_cls):>12}")
    _hbar()

    # Regression
    reg_targets = {
        "unit_weight": "Unit Weight (kN/m³)",
        "su_kpa":      "Su  (kPa)",
        "spt_n":       "SPT-N",
    }
    for col, label in reg_targets.items():
        print()
        print(f"  {label}")
        _hbar()
        print(f"  {'Method':<30} {'RMSE':>10} {'MAE':>10} {'N samples':>12}")
        _hbar()
        reg = results["regression"].get(col, {})
        for method in ["DWA", "RF", "XGB"]:
            key  = method.lower()
            m    = reg.get(key, {})
            rmse = m.get("rmse", np.nan)
            mae  = m.get("mae",  np.nan)
            n    = m.get("n",    "?")
            rmse_s = f"{rmse:.3f}" if not np.isnan(rmse) else "n/a"
            mae_s  = f"{mae:.3f}"  if not np.isnan(mae)  else "n/a"
            print(f"  {method:<30} {rmse_s:>10} {mae_s:>10} {str(n):>12}")
        _hbar()

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(data_path=DATA_PATH, model_dir=MODEL_DIR):
    print(f"\nLoading data from:  {data_path}")
    df = load_data(data_path)
    n_bh = df["borehole_id"].nunique()
    n_rows = len(df)
    print(f"  {n_rows} rows  |  {n_bh} boreholes")

    # Source subsets for each regression target
    if "su_method" in df.columns:
        su_df  = df[df["su_method"] == "ST"].copy()
        spt_df = df[df["su_method"].isna() | (df["su_method"] == "")].copy()
    else:
        su_df  = df.dropna(subset=["su_kpa"])
        spt_df = df.dropna(subset=["spt_n"])

    reg_sources = {
        "unit_weight": df,
        "su_kpa":      su_df,
        "spt_n":       spt_df,
    }

    # Try loading ML models (optional — DWA always runs)
    print(f"\nLoading models from: {model_dir}")
    models = _load_models(model_dir)
    ml_available = bool(models)

    results = {"classification": {}, "regression": {}}
    results["classification"]["n_samples"] = n_rows

    # ── Classification ────────────────────────────────────────────────────────
    print("\n=== Soil Layer Classification (LOO-CV) ===")

    print("  Running DWA …", end=" ", flush=True)
    yt, yp = _loo_classify_dwa(df)
    acc = _accuracy(yt, yp)
    results["classification"]["dwa"] = {"accuracy": acc}
    print(f"accuracy = {acc*100:.1f}%")

    if ml_available:
        print("  Running Random Forest …", end=" ", flush=True)
        yt, yp = _loo_classify_ml(df, models, "rf")
        acc = _accuracy(yt, yp)
        results["classification"]["rf"] = {"accuracy": acc}
        print(f"accuracy = {acc*100:.1f}%")

        print("  Running XGBoost …", end=" ", flush=True)
        yt, yp = _loo_classify_ml(df, models, "xgb")
        acc = _accuracy(yt, yp)
        results["classification"]["xgb"] = {"accuracy": acc}
        print(f"accuracy = {acc*100:.1f}%")
    else:
        print("  RF and XGBoost skipped (model files not found).")

    # ── Regression ────────────────────────────────────────────────────────────
    print("\n=== Property Regression (LOO-CV) ===")
    results["regression"] = {}

    reg_labels = {
        "unit_weight": "Unit Weight",
        "su_kpa":      "Su (kPa)",
        "spt_n":       "SPT-N",
    }

    for col, src in reg_sources.items():
        label = reg_labels[col]
        results["regression"][col] = {}
        n_valid = src[col].notna().sum() if col in src.columns else 0
        print(f"\n  {label}  (n valid = {n_valid})")

        print("    DWA  …", end=" ", flush=True)
        yt, yp = _loo_regress_dwa(src, col)
        rmse, mae, n = _rmse_mae(yt, yp)
        results["regression"][col]["dwa"] = {"rmse": rmse, "mae": mae, "n": n}
        print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  (n={n})")

        if ml_available:
            print("    RF   …", end=" ", flush=True)
            yt, yp = _loo_regress_ml(src, col, "rf")
            rmse, mae, n = _rmse_mae(yt, yp)
            results["regression"][col]["rf"] = {"rmse": rmse, "mae": mae, "n": n}
            print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  (n={n})")

            print("    XGB  …", end=" ", flush=True)
            yt, yp = _loo_regress_ml(src, col, "xgb")
            rmse, mae, n = _rmse_mae(yt, yp)
            results["regression"][col]["xgb"] = {"rmse": rmse, "mae": mae, "n": n}
            print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  (n={n})")
        else:
            print("    RF and XGBoost skipped.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Leave-One-Out CV by borehole for the soil profile app."
    )
    parser.add_argument("--data",   default=DATA_PATH,  help="Path to CSV dataset")
    parser.add_argument("--models", default=MODEL_DIR,   help="Path to models directory")
    parser.add_argument("--out",    default=os.path.join(_ROOT, "validation_results.json"),
                        help="Output JSON path (default: validation_results.json)")
    args = parser.parse_args()

    results = run_validation(data_path=args.data, model_dir=args.models)

    _print_results(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.out}")
    print("Upload this file in the app's 'Model Validation' tab to display the results.\n")


if __name__ == "__main__":
    main()
