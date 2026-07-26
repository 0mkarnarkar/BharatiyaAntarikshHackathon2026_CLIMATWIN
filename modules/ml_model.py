"""
CLIMATWIN — ML Model: Random Forest + SHAP Driver Analysis
===========================================================
Trains a Random Forest to predict LST from urban morphology features.
Uses SHAP to explain WHICH features drive heat in each neighborhood.

In production → replace with:
  - Physics-Informed Neural Network (DeepXDE / PyTorch)
  - Temporal Graph Neural Network (PyTorch Geometric)
  - Causal DAG discovery (DoWhy / pgmpy)
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "building_density",
    "green_cover_fraction",
    "impervious_surface_fraction",
    "road_density",
    "water_proximity",
    "pop_density_k",
    "income_index",
    "industrial",
]

FEATURE_LABELS = {
    "building_density":           "Building Density",
    "green_cover_fraction":       "Green Cover Fraction",
    "impervious_surface_fraction":"Impervious Surface",
    "road_density":               "Road Density",
    "water_proximity":            "Water Proximity",
    "pop_density_k":              "Population Density",
    "income_index":               "Income Level",
    "industrial":                 "Industrial Land Use",
}

TARGET = "lst"


def train_model(df: pd.DataFrame):
    """
    Train Random Forest on all neighborhood data.
    Returns model, scaler, SHAP explainer, SHAP values, metrics dict.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

    # Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )

    # Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Metrics
    y_pred      = model.predict(X_test)
    mae         = mean_absolute_error(y_test, y_pred)
    r2          = r2_score(y_test, y_pred)
    cv_scores   = cross_val_score(model, X_scaled_df, y, cv=5, scoring="r2")

    metrics = {
        "MAE"        : round(mae, 3),
        "R²"         : round(r2, 3),
        "CV R² Mean" : round(cv_scores.mean(), 3),
        "CV R² Std"  : round(cv_scores.std(), 3),
        "Train Size" : len(X_train),
        "Test Size"  : len(X_test),
    }

    # SHAP explainer — TreeExplainer is fast & exact for RF
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    # Attach predictions back to full df
    df = df.copy()
    df["lst_predicted"] = model.predict(X_scaled_df)
    df["residual"]      = df["lst"] - df["lst_predicted"]

    return model, scaler, explainer, shap_values, X_scaled_df, metrics, df


def get_shap_summary(shap_values, feature_names=FEATURES) -> pd.DataFrame:
    """
    Returns mean absolute SHAP values per feature — the global driver ranking.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({
        "feature"    : feature_names,
        "label"      : [FEATURE_LABELS[f] for f in feature_names],
        "shap_impact": mean_abs_shap,
    }).sort_values("shap_impact", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    summary["pct"]  = (summary["shap_impact"] / summary["shap_impact"].sum() * 100).round(1)
    return summary


def get_neighborhood_shap(shap_values, df: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    """
    Returns per-feature SHAP contributions for a single neighborhood.
    Positive SHAP = this feature is HEATING the neighborhood.
    Negative SHAP = this feature is COOLING the neighborhood.
    """
    idx = df[df["neighborhood"] == neighborhood].index
    if len(idx) == 0:
        return pd.DataFrame()
    i = idx[0]
    local_shap = shap_values[i]
    return pd.DataFrame({
        "feature"    : FEATURES,
        "label"      : [FEATURE_LABELS[f] for f in FEATURES],
        "shap_value" : local_shap,
        "direction"  : ["Heating ↑" if v > 0 else "Cooling ↓" for v in local_shap],
    }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_generator import generate_data
    df = generate_data()
    model, scaler, explainer, shap_values, X_scaled, metrics, df_pred = train_model(df)
    print("=== Model Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Global Driver Ranking (SHAP) ===")
    print(get_shap_summary(shap_values).to_string(index=False))
    print("\n=== Dharavi Local SHAP ===")
    print(get_neighborhood_shap(shap_values, df_pred, "Dharavi").to_string(index=False))
