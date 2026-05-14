import pandas as pd
import numpy as np

# Soil layers treated as cohesive (su-based SPT scale) vs granular (SPT density scale)
COHESIVE_LAYERS = {"VSC", "SOC", "SC", "MSC", "MG"}
GRANULAR_LAYERS = {"SS", "FS"}


def derive_consistency(su_kpa, spt_n, soil_layer):
    """
    Peck et al. 1974 Table 2.1.
    Priority: su_kpa first; fall back to spt_n if su is absent.
    Clay/cohesive SPT scale differs from sand/granular scale.
    """
    if pd.notna(su_kpa) and su_kpa > 0:
        if su_kpa < 15:
            return "Very Soft"
        elif su_kpa < 25:
            return "Soft"
        elif su_kpa < 50:
            return "Medium"
        elif su_kpa < 100:
            return "Stiff"
        elif su_kpa < 200:
            return "Very Stiff"
        else:
            return "Hard"

    if pd.notna(spt_n):
        n = int(spt_n)
        if soil_layer in COHESIVE_LAYERS:
            if n < 2:
                return "Very Soft"
            elif n < 4:
                return "Soft"
            elif n < 8:
                return "Medium"
            elif n < 15:
                return "Stiff"
            elif n <= 30:
                return "Very Stiff"
            else:
                return "Hard"
        else:
            if n <= 4:
                return "Very Loose"
            elif n <= 10:
                return "Loose"
            elif n <= 30:
                return "Medium Dense"
            elif n <= 50:
                return "Dense"
            else:
                return "Very Dense"

    return None


def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Normalise numeric columns
    numeric_cols = [
        "depth_m", "depth_top_m", "depth_bot_m",
        "su_kpa", "spt_n", "unit_weight",
        "plasticity_idx", "liquid_limit", "plastic_limit", "water_content",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # su_kpa comes from ST or FV samples; spt_n comes from SS samples.
    # These are mutually exclusive per row — enforce it.
    has_su_method = df["su_method"].notna() & (df["su_method"] != "")
    df.loc[has_su_method, "spt_n"] = np.nan   # ST/FV rows must not have spt_n
    df.loc[~has_su_method, "su_kpa"] = np.nan  # SS rows must not have su_kpa

    # Derive or validate consistency using Peck et al. 1974
    df["consistency"] = df.apply(
        lambda r: derive_consistency(r["su_kpa"], r["spt_n"], r["soil_layer"]),
        axis=1,
    )

    # Atterberg limits are not restricted to any soil layer type — keep all values as-is
    # (they appear wherever a lab test was performed, regardless of soil classification)

    return df


def engineer_features(df):
    """Return feature matrix X and label series for each modelling target."""
    layer_dummies = pd.get_dummies(df["soil_layer"], prefix="layer")
    base = df[["easting", "northing", "depth_m"]].copy()
    X = pd.concat([base, layer_dummies], axis=1)

    targets = {
        "soil_layer": df["soil_layer"],
        "su_kpa": df.loc[df["su_method"].isin(["ST"]), "su_kpa"],   # ST only
        "spt_n": df.loc[df["su_method"].isna() | (df["su_method"] == ""), "spt_n"],  # SS only
        "unit_weight": df["unit_weight"],
        "plasticity_idx": df["plasticity_idx"],
    }
    return X, targets
