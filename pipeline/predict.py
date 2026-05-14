"""
Inference module — wraps trained models for single-point soil predictions.

Three methods
  dwa  : distance-weighted average (baseline, no model files needed)
  rf   : Random Forest (sklearn)
  xgb  : XGBoost

Returns a structured dict with predicted values and uncertainty estimates
where available (RF gives tree-ensemble std; DWA gives weighted std;
XGBoost does not expose per-prediction variance).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_clean

# ---------------------------------------------------------------------------
# Constants (must match train.py)
# ---------------------------------------------------------------------------
ALL_LAYERS     = ["MG", "SOC", "MSC", "SC", "FS", "VSC", "SS"]
STAGE2_TARGETS = ["su_kpa", "spt_n", "unit_weight", "plasticity_idx"]

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "bangkok_boring_logs.csv"
)


# ---------------------------------------------------------------------------
# Feature constructors
# ---------------------------------------------------------------------------

def _stage1_x(easting, northing, depth):
    return np.array([[easting, northing, depth]])


def _stage2_x(easting, northing, depth, soil_layer):
    base = [easting, northing, depth]
    one_hot = [int(soil_layer == lyr) for lyr in ALL_LAYERS]
    return np.array([base + one_hot])


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class SoilPredictor:
    """Load trained models once; expose a single predict() call."""

    def __init__(self, model_dir=_DEFAULT_MODEL_DIR, data_path=_DEFAULT_DATA_PATH):
        self.df = load_and_clean(data_path)
        self._m = {}
        self._load(model_dir)

    # ── model loading ───────────────────────────────────────────────────────

    def _load(self, model_dir):
        required = (
            ["label_encoder", "stage1_rf_classifier", "stage1_xgb_classifier"]
            + [f"stage2_rf_{t}"  for t in STAGE2_TARGETS]
            + [f"stage2_xgb_{t}" for t in STAGE2_TARGETS]
        )
        missing = []
        for name in required:
            path = os.path.join(model_dir, f"{name}.joblib")
            if not os.path.exists(path):
                missing.append(path)
            else:
                self._m[name] = joblib.load(path)
        if missing:
            raise RuntimeError(
                "Missing model files — run pipeline/train.py first.\n"
                + "\n".join(f"  {p}" for p in missing)
            )

    # ── DWA helpers ─────────────────────────────────────────────────────────

    def _dwa_class(self, easting, northing, depth, k=5):
        df = self.df
        dists = np.sqrt(
            (df["easting"] - easting) ** 2
            + (df["northing"] - northing) ** 2
            + (df["depth_m"] - depth) ** 2
        ).replace(0, 1e-6)
        top = df.assign(_d=dists).nsmallest(k, "_d")
        weights = 1.0 / top["_d"].values
        vote: dict[str, float] = {}
        for lyr, w in zip(top["soil_layer"].values, weights):
            vote[lyr] = vote.get(lyr, 0.0) + w
        best = max(vote, key=vote.__getitem__)
        confidence = vote[best] / sum(vote.values())
        return best, float(confidence)

    def _dwa_value(self, col, easting, northing, depth, k=5):
        sub = self.df.dropna(subset=[col])
        if len(sub) < 2:
            return None, None
        dists = np.sqrt(
            (sub["easting"] - easting) ** 2
            + (sub["northing"] - northing) ** 2
            + (sub["depth_m"] - depth) ** 2
        ).replace(0, 1e-6)
        top = sub.assign(_d=dists).nsmallest(k, "_d")
        w = 1.0 / top["_d"].values
        vals = top[col].values
        pred = float((vals * w).sum() / w.sum())
        variance = float((w * (vals - pred) ** 2).sum() / w.sum())
        return max(0.0, pred), float(np.sqrt(variance))

    # ── RF uncertainty ──────────────────────────────────────────────────────

    @staticmethod
    def _rf_reg_std(model, x):
        """Std of individual tree predictions — a natural RF uncertainty proxy."""
        preds = np.array([t.predict(x)[0] for t in model.estimators_])
        return float(preds.std())

    # ── public API ──────────────────────────────────────────────────────────

    def predict_column(
        self,
        easting: float,
        northing: float,
        depths,
        method: str = "rf",
    ) -> list[dict]:
        """Run predict() at every depth in *depths* and return the list of result dicts."""
        results = []
        for d in depths:
            r = self.predict(easting, northing, float(d), method)
            r["depth_m"] = float(d)
            results.append(r)
        return results

    def predict(self, easting: float, northing: float, depth: float,
                method: str = "rf") -> dict:
        """
        Predict soil layer and properties at (easting, northing, depth).

        Returns
        -------
        dict with keys:
            layer             : str
            layer_confidence  : float  (0–1)
            su_kpa            : float | None
            su_kpa_std        : float | None   (None for XGBoost)
            spt_n             : float | None
            spt_n_std         : float | None
            unit_weight       : float | None
            unit_weight_std   : float | None
            plasticity_idx    : float | None
            plasticity_idx_std: float | None
            method            : str
        """
        if method == "dwa":
            return self._predict_dwa(easting, northing, depth)
        if method == "rf":
            return self._predict_ml(easting, northing, depth, prefix="rf",
                                    compute_std=True)
        if method == "xgb":
            return self._predict_ml(easting, northing, depth, prefix="xgb",
                                    compute_std=False)
        raise ValueError(f"Unknown method {method!r}. Choose: dwa / rf / xgb")

    # ── method implementations ──────────────────────────────────────────────

    def _predict_dwa(self, easting, northing, depth):
        layer, conf = self._dwa_class(easting, northing, depth)
        out = {"layer": layer, "layer_confidence": conf, "method": "dwa"}
        for col in STAGE2_TARGETS:
            val, std = self._dwa_value(col, easting, northing, depth)
            out[col] = val
            out[f"{col}_std"] = std
        return out

    def _predict_ml(self, easting, northing, depth, prefix, compute_std):
        enc = self._m["label_encoder"]
        clf = self._m[f"stage1_{prefix}_classifier"]

        x1 = _stage1_x(easting, northing, depth)
        proba = clf.predict_proba(x1)[0]
        idx = int(proba.argmax())
        layer = enc.inverse_transform([idx])[0]
        conf = float(proba[idx])

        out = {"layer": layer, "layer_confidence": conf, "method": prefix}

        x2 = _stage2_x(easting, northing, depth, layer)
        for col in STAGE2_TARGETS:
            reg = self._m[f"stage2_{prefix}_{col}"]
            val = float(max(0.0, reg.predict(x2)[0]))
            std = self._rf_reg_std(reg, x2) if compute_std else None
            out[col] = val
            out[f"{col}_std"] = std

        return out
