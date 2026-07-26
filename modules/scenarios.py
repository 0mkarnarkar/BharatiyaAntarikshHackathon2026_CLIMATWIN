"""
CLIMATWIN — Cooling Scenario Simulator
=======================================
Simulates 5 evidence-based urban cooling interventions and estimates
temperature reduction using empirical coefficients from literature.

Coefficients:
  Green Roofs     : −0.09°C per 1% increase in green cover   (Santamouris 2014)
  Street Trees    : −0.14°C per 1% increase in green cover   (Bowler et al. 2010)
  Cool Roofs      : −2.30°C per 0.1 albedo increase          (Akbari et al. 2009)
  Water Features  : −1.80°C per 0.1 increase in water prox   (Völker et al. 2013)
  Urban Forest    : −0.22°C per 1% green cover increase       (Ziter et al. 2019)

In production → replace with MARL optimizer (Ray RLlib):
  Multi-agent RL finds Pareto-optimal portfolio across budget constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict

# ------------------------------------------------------------------
# Intervention definitions
# ------------------------------------------------------------------
INTERVENTIONS = {
    "green_roofs": {
        "label"      : "🌱 Green Roofs",
        "description": "Install vegetation on building rooftops",
        "param"      : "green_cover_fraction",
        "coeff"      : -0.09,     # °C per 1% GCF increase
        "unit"       : "% green cover added",
        "max_delta"  : 0.30,      # max feasible GCF increase
        "cost_per_unit": 850,     # ₹ per sqm
        "co_benefits": ["Stormwater management", "Biodiversity", "Air quality"],
    },
    "street_trees": {
        "label"      : "🌳 Street Trees",
        "description": "Plant trees along roads and footpaths",
        "param"      : "green_cover_fraction",
        "coeff"      : -0.14,
        "unit"       : "% canopy cover added",
        "max_delta"  : 0.25,
        "cost_per_unit": 2200,
        "co_benefits": ["Shade", "Air quality", "Carbon sequestration"],
    },
    "cool_roofs": {
        "label"      : "🏠 Cool Roofs",
        "description": "Apply high-albedo reflective coatings to roofs",
        "param"      : "impervious_surface_fraction",
        "coeff"      : -2.30,     # °C per 0.1 albedo increase
        "unit"       : "albedo increase (×0.1)",
        "max_delta"  : 0.40,      # max 0.4 albedo increase
        "cost_per_unit": 120,     # ₹ per sqm of roof
        "co_benefits": ["Energy savings (AC)", "Reduced glare"],
    },
    "water_features": {
        "label"      : "💧 Water Features",
        "description": "Add urban water bodies, misting, permeable paving",
        "param"      : "water_proximity",
        "coeff"      : -1.80,
        "unit"       : "water proximity increase (×0.1)",
        "max_delta"  : 0.40,
        "cost_per_unit": 45000,   # ₹ per sqm of water feature
        "co_benefits": ["Flood resilience", "Groundwater recharge"],
    },
    "urban_forest": {
        "label"      : "🌲 Urban Forest",
        "description": "Create dense mini-forests (Miyawaki method)",
        "param"      : "green_cover_fraction",
        "coeff"      : -0.22,
        "unit"       : "% forest cover added",
        "max_delta"  : 0.20,
        "cost_per_unit": 1800,
        "co_benefits": ["Biodiversity hotspot", "Carbon sink", "Mental health"],
    },
}


def simulate_single(
    row: pd.Series,
    intervention_key: str,
    delta: float
) -> Dict:
    """
    Simulate one intervention on one neighborhood.
    Returns a dict with before/after LST and cost estimate.
    """
    iv     = INTERVENTIONS[intervention_key]
    coeff  = iv["coeff"]
    delta  = min(delta, iv["max_delta"])

    # Temperature reduction estimate
    if intervention_key in ("green_roofs", "street_trees", "urban_forest"):
        # coeff is per 1% GCF increase; delta is fractional (0.10 = 10%)
        delta_temp = coeff * (delta * 100)
    else:
        # coeff is per 0.1 unit; delta is fractional
        delta_temp = coeff * (delta / 0.10)

    lst_before = row["lst"]
    lst_after  = round(lst_before + delta_temp, 2)

    # Cost estimate
    area_sqkm  = row.get("area_sqkm", 3.0)
    area_sqm   = area_sqkm * 1_000_000
    if intervention_key == "cool_roofs":
        # Only roof area (~30% of footprint)
        applicable_area = area_sqm * row["building_density"] * 0.30
    elif intervention_key == "water_features":
        applicable_area = area_sqm * delta * 0.05
    else:
        applicable_area = area_sqm * delta

    cost_lakh  = round((applicable_area * delta * iv["cost_per_unit"]) / 1e5, 1)
    cost_lakh  = min(cost_lakh, 5000.0)   # cap display

    return {
        "neighborhood"   : row["neighborhood"],
        "intervention"   : iv["label"],
        "lst_before"     : lst_before,
        "lst_after"      : lst_after,
        "delta_temp"     : round(delta_temp, 2),
        "delta_applied"  : round(delta, 3),
        "cost_lakh"      : cost_lakh,
        "cost_per_degC"  : round(cost_lakh / abs(delta_temp), 1) if delta_temp != 0 else 0,
        "co_benefits"    : iv["co_benefits"],
    }


def simulate_portfolio(
    df: pd.DataFrame,
    selections: Dict[str, float],
    target_neighborhoods: list = None,
) -> pd.DataFrame:
    """
    Simulate a portfolio of interventions across selected neighborhoods.

    Parameters
    ----------
    df                   : neighborhood DataFrame
    selections           : {intervention_key: delta_value}
    target_neighborhoods : list of neighborhood names; None = all

    Returns combined DataFrame with before/after stats.
    """
    if target_neighborhoods:
        sub = df[df["neighborhood"].isin(target_neighborhoods)].copy()
    else:
        sub = df.copy()

    results = []
    for _, row in sub.iterrows():
        combined_delta = 0.0
        costs          = 0.0
        for iv_key, delta in selections.items():
            if delta > 0:
                res = simulate_single(row, iv_key, delta)
                combined_delta += res["delta_temp"]
                costs          += res["cost_lakh"]

        results.append({
            "neighborhood"  : row["neighborhood"],
            "lat"           : row["lat"],
            "lon"           : row["lon"],
            "lst_before"    : row["lst"],
            "lst_after"     : round(max(row["lst"] + combined_delta, 24.0), 2),
            "reduction"     : round(abs(combined_delta), 2),
            "cost_lakh"     : round(costs, 1),
            "equity_score"  : row.get("equity_score", 50),
            "pop_density_k" : row.get("pop_density_k", 10),
        })

    return pd.DataFrame(results).sort_values("reduction", ascending=False)


def rank_by_priority(
    portfolio_df: pd.DataFrame,
    weight_equity: float = 0.5,
    weight_cooling: float = 0.3,
    weight_cost: float = 0.2,
) -> pd.DataFrame:
    """
    Rank neighborhoods by intervention priority:
      Priority = w_eq · equity_score_norm
               + w_cool · cooling_potential_norm
               - w_cost · cost_norm

    Higher score = intervene here first.
    """
    df = portfolio_df.copy()
    df["eq_norm"]   = df["equity_score"]   / 100
    df["cool_norm"] = df["reduction"]      / df["reduction"].max()
    max_cost        = df["cost_lakh"].replace(0, np.nan).max()
    df["cost_norm"] = df["cost_lakh"]      / (max_cost if max_cost else 1)

    df["priority_score"] = (
        weight_equity  * df["eq_norm"]
        + weight_cooling * df["cool_norm"]
        - weight_cost    * df["cost_norm"]
    )
    df["priority_score"]  = (df["priority_score"] - df["priority_score"].min())
    df["priority_score"] /= (df["priority_score"].max() + 1e-9)
    df["priority_score"]  = (df["priority_score"] * 100).round(1)

    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from modules.data_generator import generate_data

    df = generate_data()

    # Example: simulate green roofs + cool roofs portfolio
    selections = {"green_roofs": 0.15, "cool_roofs": 0.20}
    results    = simulate_portfolio(df, selections)
    ranked     = rank_by_priority(results)

    print("=== Top 10 Priority Interventions ===")
    print(ranked[["neighborhood","lst_before","lst_after","reduction",
                   "cost_lakh","priority_score"]].head(10).to_string(index=False))
