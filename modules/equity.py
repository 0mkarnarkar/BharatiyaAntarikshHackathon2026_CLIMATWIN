"""
CLIMATWIN — Thermal Equity Index
==================================
Identifies communities that face BOTH high heat exposure AND
high social vulnerability — the most urgent intervention targets.

Equity Score = f(LST, population density, income level, green cover deficit)

Research basis:
  Harlan et al. (2006): Heat mortality concentrated in low-income,
  low-vegetation neighborhoods (Phoenix study).
  District-level Indian mortality data: top 41% high-burden districts
  bear 57% of heatwave excess deaths (Azhar et al. 2014).
"""

import numpy as np
import pandas as pd


INCOME_LABELS = {
    (0.00, 0.20): "Extreme Poverty",
    (0.20, 0.40): "Low Income",
    (0.40, 0.60): "Lower-Middle",
    (0.60, 0.80): "Middle Income",
    (0.80, 1.00): "High Income",
}


def income_label(val: float) -> str:
    for (lo, hi), label in INCOME_LABELS.items():
        if lo <= val < hi:
            return label
    return "High Income"


def compute_equity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a multi-dimensional thermal equity index for each neighborhood.

    Dimensions:
      1. Heat Exposure Score    (0–1) — normalized LST
      2. Vulnerability Score    (0–1) — low income + high density
      3. Adaptive Capacity      (0–1) — green access + income (inverse vulnerability)
      4. Green Cover Deficit    (0–1) — 1 − green_cover vs city mean
      5. Composite Equity Score (0–100) — weighted combination

    Quadrant classification:
      Q1 Hot + Vulnerable   → URGENT (red)
      Q2 Hot + Resilient    → MONITOR (orange)
      Q3 Cool + Vulnerable  → SUPPORT (yellow)
      Q4 Cool + Resilient   → STABLE  (green)
    """
    df = df.copy()

    # --- 1. Heat Exposure Score (0–1) ---
    lst_min, lst_max = df["lst"].min(), df["lst"].max()
    df["heat_exposure"] = (df["lst"] - lst_min) / (lst_max - lst_min)

    # --- 2. Vulnerability Score (0–1) ---
    # Low income + high density = more vulnerable
    pop_norm = np.clip(df["pop_density_k"] / 300.0, 0, 1)
    df["vulnerability"] = 0.55 * (1 - df["income_index"]) + 0.45 * pop_norm

    # --- 3. Adaptive Capacity (0–1, higher = more resilient) ---
    df["adaptive_capacity"] = (
        0.40 * df["income_index"]
        + 0.35 * df["green_cover_fraction"]
        + 0.25 * df["water_proximity"]
    )

    # --- 4. Green Cover Deficit ---
    mean_green = df["green_cover_fraction"].mean()
    df["green_deficit"] = np.clip(mean_green - df["green_cover_fraction"], 0, 1)

    # --- 5. Composite Equity Score (0–100) ---
    df["equity_score"] = (
        0.40 * df["heat_exposure"]
        + 0.35 * df["vulnerability"]
        + 0.15 * df["green_deficit"]
        - 0.10 * df["adaptive_capacity"]
    )
    df["equity_score"] = (
        (df["equity_score"] - df["equity_score"].min())
        / (df["equity_score"].max() - df["equity_score"].min() + 1e-9)
        * 100
    ).round(1)

    # --- Quadrant ---
    heat_median  = df["heat_exposure"].median()
    vuln_median  = df["vulnerability"].median()

    def classify(row):
        hot  = row["heat_exposure"]  >= heat_median
        vuln = row["vulnerability"]  >= vuln_median
        if hot and vuln:
            return "🔴 Urgent"
        elif hot and not vuln:
            return "🟠 Monitor"
        elif not hot and vuln:
            return "🟡 Support"
        else:
            return "🟢 Stable"

    df["equity_class"]  = df.apply(classify, axis=1)
    df["income_label"]  = df["income_index"].apply(income_label)

    # --- Intervention Priority Text ---
    def priority_text(row):
        if row["equity_class"] == "🔴 Urgent":
            return "Immediate street trees + cool roofs + equity funding"
        elif row["equity_class"] == "🟠 Monitor":
            return "Cool roofs + water features; voluntary greening program"
        elif row["equity_class"] == "🟡 Support":
            return "Social welfare + proactive greening investment"
        else:
            return "Maintain standards; expand green to buffer zones"

    df["intervention_note"] = df.apply(priority_text, axis=1)

    return df


def city_summary(df: pd.DataFrame) -> dict:
    """
    High-level equity summary statistics for the dashboard header.
    """
    eq = compute_equity_index(df)
    urgent_pop = eq[eq["equity_class"] == "🔴 Urgent"]["pop_density_k"].sum()
    total_pop  = eq["pop_density_k"].sum()

    return {
        "urgent_zones"      : int((eq["equity_class"] == "🔴 Urgent").sum()),
        "total_zones"       : len(eq),
        "urgent_pop_pct"    : round(urgent_pop / total_pop * 100, 1) if total_pop else 0,
        "mean_equity_score" : round(eq["equity_score"].mean(), 1),
        "max_equity_score"  : round(eq["equity_score"].max(), 1),
        "hottest_vulnerable": eq.loc[eq["equity_score"].idxmax(), "neighborhood"],
        "green_deficit_mean": round(eq["green_deficit"].mean() * 100, 1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from modules.data_generator import generate_data

    df   = generate_data()
    eq   = compute_equity_index(df)
    summ = city_summary(df)

    print("=== Equity Summary ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")

    print("\n=== Urgent Zones ===")
    urgent = eq[eq["equity_class"] == "🔴 Urgent"][[
        "neighborhood", "lst", "equity_score", "income_label", "pop_density_k"
    ]].sort_values("equity_score", ascending=False)
    print(urgent.to_string(index=False))
