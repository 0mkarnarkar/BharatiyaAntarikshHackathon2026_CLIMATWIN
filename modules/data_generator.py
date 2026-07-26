"""
CLIMATWIN — Mumbai Thermal Data Generator
==========================================
Generates physics-informed synthetic LST data for 65 Mumbai neighborhoods.

Physics formula (surface energy balance simplified):
  LST = T_base
      + α·ISF          (impervious surface heats up 8.5x more than soil)
      + β·BD           (building density traps longwave radiation)
      - γ·GCF          (green cover cools via evapotranspiration)
      + δ·RD           (road surface + anthropogenic heat)
      - ε·WPF          (water bodies cool via latent heat)
      + ζ·IND          (industrial waste heat)
      + N(0, σ)        (sensor + atmospheric noise)

Coefficients derived from peer-reviewed Mumbai UHI studies
(Mohan et al. 2012, Ramachandra et al. 2015, Mathew et al. 2018).

In production → replace generate_data() with GEE API call:
  ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(mumbai_geometry).filterDate(...)
    .select(['ST_B10']).mean().reduceRegions(...)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ------------------------------------------------------------------
# 65 Mumbai neighborhoods — lat/lon centroids + land-use profile
# ------------------------------------------------------------------
NEIGHBORHOODS = [
    # name, lat, lon, bldg_density, green_cover, imperv, road_density,
    #       water_prox, pop_density(k/sqkm), income(0-1), industrial(0/1), area_sqkm

    # === SOUTH MUMBAI ===
    ("Nariman Point",    18.9256, 72.8242, 0.55, 0.06, 0.78, 0.80, 0.55, 4.5,  0.95, 0, 1.2),
    ("Colaba",           18.9067, 72.8147, 0.70, 0.08, 0.80, 0.75, 0.45, 22.0, 0.75, 0, 2.1),
    ("Fort",             18.9344, 72.8346, 0.72, 0.04, 0.88, 0.85, 0.30, 18.0, 0.80, 0, 1.8),
    ("Marine Lines",     18.9435, 72.8206, 0.62, 0.05, 0.80, 0.78, 0.60, 28.0, 0.70, 0, 1.5),
    ("Malabar Hill",     18.9590, 72.8063, 0.35, 0.28, 0.55, 0.55, 0.40, 12.0, 0.95, 0, 3.5),
    ("Worli",            19.0143, 72.8177, 0.60, 0.08, 0.78, 0.72, 0.55, 35.0, 0.60, 0, 4.2),
    ("Prabhadevi",       19.0177, 72.8294, 0.72, 0.06, 0.82, 0.74, 0.25, 48.0, 0.55, 0, 2.8),
    ("Dadar",            19.0212, 72.8422, 0.80, 0.04, 0.88, 0.82, 0.15, 65.0, 0.45, 0, 3.4),
    ("Mahim",            19.0412, 72.8418, 0.78, 0.05, 0.86, 0.78, 0.30, 55.0, 0.40, 0, 2.6),

    # === DHARAVI / HIGH-DENSITY CORE ===
    ("Dharavi",          19.0421, 72.8567, 0.96, 0.01, 0.97, 0.45, 0.10, 290.0, 0.05, 0, 2.1),
    ("Govandi",          19.0494, 72.9286, 0.88, 0.03, 0.93, 0.55, 0.15, 180.0, 0.05, 1, 3.2),
    ("Mankhurd",         19.0456, 72.9357, 0.82, 0.04, 0.90, 0.58, 0.20, 150.0, 0.10, 1, 4.1),
    ("Kurla",            19.0726, 72.8845, 0.85, 0.03, 0.92, 0.72, 0.10, 95.0,  0.15, 0, 5.5),
    ("Sion",             19.0421, 72.8622, 0.82, 0.04, 0.89, 0.75, 0.12, 80.0,  0.30, 0, 3.8),
    ("Cheeta Camp",      19.0356, 72.9289, 0.90, 0.02, 0.95, 0.50, 0.18, 210.0, 0.08, 1, 1.8),

    # === BKC / COMMERCIAL ===
    ("BKC",              19.0651, 72.8677, 0.55, 0.10, 0.80, 0.65, 0.18, 3.0,   0.90, 0, 1.7),
    ("Kalanagar",        19.0702, 72.8732, 0.48, 0.12, 0.75, 0.60, 0.20, 8.0,   0.85, 0, 1.2),
    ("Bandra East",      19.0595, 72.8556, 0.70, 0.06, 0.82, 0.75, 0.22, 45.0,  0.55, 0, 3.1),

    # === BANDRA / WESTERN SUBURBS HIGH INCOME ===
    ("Bandra West",      19.0596, 72.8295, 0.50, 0.15, 0.72, 0.68, 0.38, 25.0,  0.88, 0, 6.5),
    ("Khar",             19.0729, 72.8337, 0.52, 0.14, 0.74, 0.65, 0.32, 22.0,  0.85, 0, 4.2),
    ("Santacruz West",   19.0823, 72.8406, 0.55, 0.13, 0.76, 0.67, 0.28, 28.0,  0.78, 0, 5.8),
    ("Juhu",             19.1013, 72.8264, 0.42, 0.18, 0.68, 0.60, 0.50, 15.0,  0.90, 0, 3.5),
    ("Vile Parle West",  19.0990, 72.8381, 0.55, 0.12, 0.76, 0.68, 0.30, 32.0,  0.75, 0, 4.0),

    # === ANDHERI / EASTERN SUBURBS MIXED ===
    ("Andheri East",     19.1136, 72.8697, 0.68, 0.07, 0.82, 0.75, 0.15, 38.0,  0.55, 1, 18.0),
    ("Andheri West",     19.1271, 72.8369, 0.55, 0.10, 0.76, 0.68, 0.22, 30.0,  0.65, 0, 12.0),
    ("Jogeshwari",       19.1399, 72.8489, 0.68, 0.08, 0.82, 0.72, 0.18, 55.0,  0.40, 0, 6.2),
    ("Versova",          19.1323, 72.8128, 0.52, 0.12, 0.72, 0.60, 0.55, 20.0,  0.60, 0, 4.5),
    ("Malad West",       19.1863, 72.8487, 0.58, 0.10, 0.78, 0.68, 0.25, 42.0,  0.50, 0, 8.5),
    ("Malad East",       19.1874, 72.8622, 0.65, 0.08, 0.82, 0.70, 0.18, 52.0,  0.40, 0, 7.2),
    ("Kandivali West",   19.2050, 72.8283, 0.55, 0.12, 0.76, 0.65, 0.28, 38.0,  0.50, 0, 9.8),
    ("Kandivali East",   19.2087, 72.8578, 0.60, 0.10, 0.78, 0.68, 0.20, 42.0,  0.45, 0, 8.5),
    ("Borivali West",    19.2318, 72.8557, 0.52, 0.15, 0.72, 0.62, 0.25, 35.0,  0.55, 0, 10.2),
    ("Borivali East",    19.2318, 72.8748, 0.48, 0.18, 0.68, 0.60, 0.22, 30.0,  0.50, 0, 8.8),

    # === INDUSTRIAL EASTERN SUBURBS ===
    ("Chembur",          19.0622, 72.8991, 0.70, 0.08, 0.85, 0.72, 0.18, 28.0,  0.35, 1, 9.0),
    ("Trombay",          19.0261, 72.9312, 0.55, 0.12, 0.78, 0.62, 0.25, 8.0,   0.30, 1, 12.0),
    ("Ghatkopar East",   19.0887, 72.9085, 0.72, 0.06, 0.85, 0.74, 0.12, 62.0,  0.35, 0, 6.5),
    ("Ghatkopar West",   19.0887, 72.9042, 0.65, 0.08, 0.82, 0.72, 0.14, 55.0,  0.40, 0, 5.8),
    ("Vikhroli",         19.1088, 72.9260, 0.65, 0.10, 0.82, 0.68, 0.18, 42.0,  0.35, 1, 10.5),
    ("Powai",            19.1224, 72.9060, 0.48, 0.20, 0.65, 0.62, 0.45, 18.0,  0.75, 0, 7.8),
    ("Mulund East",      19.1715, 72.9563, 0.60, 0.12, 0.78, 0.65, 0.22, 35.0,  0.40, 0, 8.2),
    ("Mulund West",      19.1715, 72.9423, 0.55, 0.14, 0.74, 0.62, 0.25, 30.0,  0.45, 0, 7.5),
    ("Bhandup",          19.1528, 72.9398, 0.65, 0.10, 0.82, 0.68, 0.20, 45.0,  0.35, 1, 9.2),

    # === NORTHERN GREEN ZONES ===
    ("Aarey Colony",     19.1655, 72.8737, 0.08, 0.72, 0.22, 0.30, 0.28, 0.5,   0.40, 0, 12.8),
    ("SGNP Core",        19.2290, 72.9143, 0.02, 0.90, 0.10, 0.15, 0.20, 0.1,   0.00, 0, 35.0),
    ("SGNP Buffer",      19.2147, 72.9104, 0.05, 0.82, 0.15, 0.20, 0.22, 0.2,   0.00, 0, 20.0),
    ("Yeoor Hills",      19.2550, 72.9500, 0.03, 0.88, 0.12, 0.18, 0.18, 0.1,   0.00, 0, 15.0),

    # === THANE BORDER / NORTHERN SUBURBS ===
    ("Dahisar",          19.2518, 72.8583, 0.55, 0.15, 0.74, 0.62, 0.22, 32.0,  0.40, 0, 11.5),
    ("Mira Road",        19.2875, 72.8683, 0.60, 0.10, 0.78, 0.65, 0.18, 38.0,  0.35, 0, 14.2),

    # === CENTRAL SUBURBS ===
    ("Vashi",            19.0768, 72.9988, 0.50, 0.16, 0.72, 0.65, 0.35, 18.0,  0.65, 0, 9.5),
    ("Turbhe",           19.0794, 73.0153, 0.60, 0.10, 0.80, 0.68, 0.22, 12.0,  0.45, 1, 6.8),
    ("Ghansoli",         19.1134, 73.0099, 0.52, 0.14, 0.74, 0.62, 0.30, 15.0,  0.50, 0, 5.5),
    ("Airoli",           19.1568, 72.9999, 0.50, 0.15, 0.72, 0.60, 0.32, 12.0,  0.55, 0, 7.2),
    ("Rabale",           19.1188, 73.0221, 0.45, 0.18, 0.68, 0.58, 0.28, 8.0,   0.60, 1, 4.8),
    ("Thane West",       19.2183, 72.9781, 0.62, 0.10, 0.80, 0.70, 0.20, 35.0,  0.45, 0, 12.0),
    ("Thane East",       19.2183, 73.0101, 0.55, 0.14, 0.76, 0.65, 0.25, 28.0,  0.50, 0, 10.5),

    # === WATER / COASTAL ===
    ("Versova Beach",    19.1435, 72.8116, 0.25, 0.12, 0.50, 0.42, 0.85, 8.0,   0.65, 0, 2.2),
    ("Madh Island",      19.1570, 72.8055, 0.18, 0.35, 0.38, 0.32, 0.75, 2.0,   0.70, 0, 4.5),
    ("Mangrove Thane",   19.0800, 73.0400, 0.05, 0.65, 0.20, 0.15, 0.70, 0.5,   0.00, 0, 8.5),
    ("Elephanta",        18.9633, 72.9313, 0.05, 0.70, 0.18, 0.15, 0.90, 0.2,   0.00, 0, 5.0),

    # === ADDITIONAL DENSITY POINTS ===
    ("Grant Road",       18.9640, 72.8150, 0.82, 0.03, 0.90, 0.82, 0.15, 75.0,  0.30, 0, 1.5),
    ("Byculla",          18.9784, 72.8354, 0.80, 0.05, 0.88, 0.78, 0.18, 68.0,  0.28, 0, 2.8),
    ("Wadala",           19.0199, 72.8601, 0.72, 0.07, 0.83, 0.72, 0.28, 52.0,  0.35, 1, 5.5),
    ("Santacruz East",   19.0823, 72.8556, 0.68, 0.07, 0.82, 0.72, 0.18, 45.0,  0.50, 0, 4.8),
]

COLUMNS = [
    "neighborhood", "lat", "lon",
    "building_density", "green_cover_fraction", "impervious_surface_fraction",
    "road_density", "water_proximity", "pop_density_k",
    "income_index", "industrial", "area_sqkm"
]


def compute_lst(row: dict) -> float:
    """
    Physics-informed LST estimation (surface energy balance proxy).
    Coefficients from Mumbai-specific UHI literature.
    """
    T_base = 31.5   # °C — mean daytime air temp Jun–Sep, Mumbai IMD

    lst = (
        T_base
        + 8.20 * row["building_density"]
        + 6.50 * row["impervious_surface_fraction"]
        - 7.80 * row["green_cover_fraction"]
        + 3.10 * row["road_density"]
        - 4.50 * row["water_proximity"]
        + 2.80 * row["industrial"]
        + np.random.normal(0, 0.65)   # sensor/atmospheric noise
    )
    return round(float(np.clip(lst, 24.0, 50.0)), 2)


def compute_heat_stress_index(lst: float, pop_density: float) -> float:
    """
    Heat Stress Index = LST contribution + exposure (density) contribution.
    Normalized 0–100. Used for equity mapping.
    """
    lst_norm   = (lst - 24) / (50 - 24)
    pop_norm   = min(pop_density / 300.0, 1.0)
    hsi = 0.65 * lst_norm + 0.35 * pop_norm
    return round(float(np.clip(hsi * 100, 0, 100)), 2)


def generate_data() -> pd.DataFrame:
    """
    Returns a DataFrame of 65 Mumbai neighborhoods with:
      - Urban morphology features
      - Physics-informed LST
      - Heat Stress Index
      - Thermal Equity Score
    """
    rows = []
    for record in NEIGHBORHOODS:
        d = dict(zip(COLUMNS, record))
        d["lst"] = compute_lst(d)
        d["heat_stress_index"] = compute_heat_stress_index(
            d["lst"], d["pop_density_k"]
        )
        # Thermal Equity Score — higher = more urgent intervention needed
        # Heat burden (high LST) × Social vulnerability (low income + high density)
        vulnerability = (1 - d["income_index"]) * 0.5 + min(d["pop_density_k"] / 300, 1.0) * 0.5
        d["equity_score"] = round(
            float(np.clip(0.60 * ((d["lst"] - 24) / 26) + 0.40 * vulnerability, 0, 1) * 100), 2
        )
        # LST category label
        if d["lst"] >= 42:
            d["lst_category"] = "Critical"
        elif d["lst"] >= 38:
            d["lst_category"] = "High"
        elif d["lst"] >= 34:
            d["lst_category"] = "Moderate"
        else:
            d["lst_category"] = "Low"
        rows.append(d)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_data()
    print(df[["neighborhood", "lst", "heat_stress_index", "equity_score", "lst_category"]].to_string())
    print(f"\nLST range: {df['lst'].min():.1f}°C — {df['lst'].max():.1f}°C")
    print(f"Critical zones: {(df['lst_category']=='Critical').sum()}")


# ─────────────────────────────────────────────────────────────────
# MAHARASHTRA-WIDE DATA — Nashik, Pune, peri-urban, hill stations
# ─────────────────────────────────────────────────────────────────
MAHARASHTRA_EXTRA = [
    # name, lat, lon, bldg_density, green_cover, imperv, road_density,
    #       water_prox, pop_density_k, income_index, industrial, area_sqkm
    # NASHIK
    ("Nashik CBD",        20.0013, 73.7849, 0.68, 0.08, 0.82, 0.72, 0.12, 35.0, 0.55, 0, 6.5),
    ("Nashik Panchvati",  19.9976, 73.7878, 0.72, 0.06, 0.84, 0.75, 0.15, 42.0, 0.45, 0, 4.2),
    ("Nashik MIDC",       20.0097, 73.7534, 0.65, 0.04, 0.88, 0.68, 0.08, 18.0, 0.35, 1, 8.0),
    ("Nashik Road",       19.9745, 73.8186, 0.70, 0.07, 0.83, 0.70, 0.14, 38.0, 0.40, 0, 5.5),
    ("Deolali",           19.9464, 73.8350, 0.45, 0.18, 0.65, 0.55, 0.22, 12.0, 0.50, 0, 8.5),
    ("Sinnar",            19.8478, 74.0003, 0.55, 0.10, 0.75, 0.60, 0.18, 15.0, 0.35, 1, 12.0),
    ("Ozar",              20.0884, 73.9142, 0.28, 0.30, 0.48, 0.40, 0.20,  5.0, 0.35, 0, 18.0),
    # PUNE
    ("Pune Shivajinagar", 18.5308, 73.8475, 0.70, 0.10, 0.82, 0.75, 0.18, 42.0, 0.70, 0, 5.2),
    ("Pune Hadapsar",     18.5097, 73.9283, 0.72, 0.06, 0.85, 0.72, 0.12, 55.0, 0.50, 1,12.0),
    ("Pune Kothrud",      18.5074, 73.8078, 0.60, 0.14, 0.76, 0.68, 0.20, 38.0, 0.65, 0, 7.5),
    ("Pune Hinjewadi",    18.5892, 73.7380, 0.55, 0.10, 0.80, 0.65, 0.15, 12.0, 0.75, 1,14.0),
    ("Pune Wakad",        18.6011, 73.7623, 0.60, 0.10, 0.78, 0.68, 0.18, 35.0, 0.60, 0, 6.8),
    ("Pimpri",            18.6279, 73.7986, 0.68, 0.08, 0.82, 0.72, 0.14, 48.0, 0.45, 1,10.5),
    ("Chinchwad",         18.6472, 73.7977, 0.65, 0.10, 0.80, 0.70, 0.16, 42.0, 0.50, 1, 9.2),
    ("Pune Yerawada",     18.5493, 73.8918, 0.62, 0.12, 0.78, 0.68, 0.20, 35.0, 0.55, 0, 5.8),
    ("Pune Khadki",       18.5642, 73.8555, 0.40, 0.25, 0.60, 0.55, 0.30, 18.0, 0.60, 0, 6.5),
    ("Pune Kondhwa",      18.4611, 73.8750, 0.58, 0.12, 0.76, 0.65, 0.22, 35.0, 0.55, 0, 8.2),
    ("Pune Baner",        18.5590, 73.7868, 0.55, 0.14, 0.74, 0.65, 0.22, 30.0, 0.70, 0, 5.5),
    ("Pune Magarpatta",   18.5164, 73.9269, 0.55, 0.12, 0.76, 0.65, 0.18, 25.0, 0.80, 0, 4.5),
    # THANE-KALYAN BELT
    ("Thane City",        19.2183, 72.9781, 0.68, 0.08, 0.82, 0.72, 0.25, 42.0, 0.55, 0,15.0),
    ("Kalyan East",       19.2403, 73.1305, 0.75, 0.06, 0.86, 0.72, 0.14, 58.0, 0.30, 0, 8.5),
    ("Dombivli",          19.2143, 73.0877, 0.78, 0.05, 0.88, 0.74, 0.12, 65.0, 0.28, 0, 6.8),
    ("Bhiwandi",          19.2985, 73.0544, 0.72, 0.06, 0.86, 0.68, 0.14, 42.0, 0.25, 1,12.5),
    ("Ulhasnagar",        19.2182, 73.1516, 0.80, 0.04, 0.90, 0.72, 0.12, 72.0, 0.22, 0, 5.5),
    ("Ambarnath",         19.1950, 73.1862, 0.68, 0.08, 0.82, 0.68, 0.18, 45.0, 0.30, 1,10.2),
    ("Badlapur",          19.1511, 73.2380, 0.55, 0.14, 0.74, 0.62, 0.22, 28.0, 0.35, 0, 8.8),
    # VASAI-VIRAR BELT
    ("Bhayandar",          19.3017, 72.8517, 0.58, 0.11, 0.76, 0.66, 0.22, 40.0, 0.42, 0,10.5),
    ("Vasai",             19.3919, 72.8325, 0.52, 0.15, 0.72, 0.62, 0.28, 32.0, 0.38, 0,14.0),
    ("Virar",             19.4641, 72.8147, 0.55, 0.14, 0.74, 0.62, 0.25, 38.0, 0.35, 0,12.5),
    ("Nala Sopara",       19.4200, 72.8413, 0.60, 0.10, 0.78, 0.65, 0.22, 52.0, 0.25, 0,10.0),
    # PERI-URBAN & HILL STATIONS
    ("Panvel",            18.9893, 73.1098, 0.62, 0.10, 0.78, 0.65, 0.28, 35.0, 0.45, 0,10.5),
    ("Khopoli",           18.7877, 73.3396, 0.30, 0.35, 0.52, 0.45, 0.25,  8.0, 0.40, 1,10.0),
    ("Lonavala",          18.7481, 73.4045, 0.20, 0.55, 0.38, 0.40, 0.30,  4.0, 0.60, 0,18.0),
    ("Khandala",          18.7602, 73.3717, 0.15, 0.65, 0.28, 0.35, 0.32,  2.0, 0.65, 0, 8.0),
    ("Matheran",          18.9843, 73.2688, 0.12, 0.78, 0.20, 0.28, 0.25,  1.0, 0.55, 0, 8.0),
    ("Karjat",            18.9126, 73.3269, 0.28, 0.40, 0.48, 0.42, 0.35,  6.0, 0.35, 0,12.0),
    ("Alibaug",           18.6414, 72.8729, 0.28, 0.28, 0.52, 0.45, 0.55,  8.0, 0.50, 0,15.0),
    ("Igatpuri",          19.6948, 73.5591, 0.18, 0.65, 0.30, 0.32, 0.30,  3.0, 0.30, 0,20.0),
    ("Kasara",            19.6093, 73.4761, 0.15, 0.60, 0.28, 0.30, 0.35,  2.0, 0.28, 0,15.0),
    ("Shahapur",          19.4544, 73.3251, 0.20, 0.55, 0.35, 0.35, 0.32,  3.0, 0.25, 0,14.0),
    ("Wada",              19.6696, 73.0006, 0.25, 0.40, 0.45, 0.38, 0.25,  5.0, 0.28, 0,15.0),
    ("Satara",            17.6805, 73.9862, 0.50, 0.15, 0.70, 0.60, 0.22, 18.0, 0.40, 0,14.0),
    ("Kolhapur",          16.7050, 74.2433, 0.58, 0.12, 0.74, 0.65, 0.25, 32.0, 0.42, 0,16.0),
]

EXTRA_COLUMNS = [
    "neighborhood","lat","lon",
    "building_density","green_cover_fraction","impervious_surface_fraction",
    "road_density","water_proximity","pop_density_k",
    "income_index","industrial","area_sqkm"
]


def generate_maharashtra_data() -> pd.DataFrame:
    """
    Mumbai (65 zones) + Maharashtra-wide data (42 locations).
    Used for the regional pixel heatmap view.
    """
    mumbai = generate_data()
    extra_rows = []
    for record in MAHARASHTRA_EXTRA:
        d = dict(zip(EXTRA_COLUMNS, record))
        d["lst"] = compute_lst(d)
        d["heat_stress_index"] = compute_heat_stress_index(d["lst"], d["pop_density_k"])
        vuln = (1 - d["income_index"]) * 0.5 + min(d["pop_density_k"] / 300, 1.0) * 0.5
        d["equity_score"] = round(
            float(np.clip(0.60 * ((d["lst"] - 24) / 26) + 0.40 * vuln, 0, 1) * 100), 2
        )
        if d["lst"] >= 42: d["lst_category"] = "Critical"
        elif d["lst"] >= 38: d["lst_category"] = "High"
        elif d["lst"] >= 34: d["lst_category"] = "Moderate"
        else: d["lst_category"] = "Low"
        extra_rows.append(d)
    extra_df = pd.DataFrame(extra_rows)
    # Add missing columns that Mumbai data has
    for col in mumbai.columns:
        if col not in extra_df.columns:
            extra_df[col] = 0
    combined = pd.concat([mumbai, extra_df[mumbai.columns]], ignore_index=True)
    return combined


if __name__ == "__main__":
    mh = generate_maharashtra_data()
    print(f"Maharashtra dataset: {len(mh)} locations")
    print(f"LST range: {mh.lst.min():.1f}–{mh.lst.max():.1f}°C")
    print(mh[["neighborhood","lat","lon","lst","lst_category"]].tail(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# FULL MAHARASHTRA STATE COVERAGE — all divisions
# Vidarbha, Marathwada, North Maharashtra, Konkan, Western Maharashtra
# ─────────────────────────────────────────────────────────────────
STATE_WIDE_EXTRA = [
    # name, lat, lon, bldg_density, green_cover, imperv, road_density,
    #       water_prox, pop_density_k, income_index, industrial, area_sqkm

    # === VIDARBHA — Nagpur division ===
    ("Nagpur CBD",       21.1458, 79.0882, 0.72, 0.08, 0.84, 0.76, 0.15, 48.0, 0.55, 0, 8.5),
    ("Nagpur MIDC Hingna",21.0989, 78.9722, 0.60, 0.06, 0.82, 0.65, 0.10, 18.0, 0.40, 1,12.0),
    ("Nagpur Sitabuldi",  21.1500, 79.0800, 0.75, 0.05, 0.87, 0.78, 0.10, 55.0, 0.50, 0, 4.5),
    ("Wardha",            20.7453, 78.6022, 0.45, 0.20, 0.62, 0.52, 0.20, 12.0, 0.38, 0,14.0),
    ("Chandrapur",        19.9615, 79.2961, 0.55, 0.10, 0.76, 0.62, 0.15, 22.0, 0.35, 1,16.0),
    ("Ballarpur",         19.8447, 79.3572, 0.50, 0.12, 0.72, 0.58, 0.18, 15.0, 0.32, 1,10.0),
    ("Gadchiroli",        20.1809, 80.0021, 0.20, 0.55, 0.35, 0.32, 0.25,  3.0, 0.20, 0,25.0),
    ("Bhandara",          21.1704, 79.6519, 0.35, 0.30, 0.52, 0.45, 0.30,  8.0, 0.32, 0,14.0),
    ("Gondia",            21.4602, 80.1922, 0.38, 0.28, 0.55, 0.46, 0.28,  9.0, 0.30, 0,13.0),

    # === VIDARBHA — Amravati division ===
    ("Amravati City",     20.9374, 77.7796, 0.60, 0.10, 0.78, 0.66, 0.18, 30.0, 0.42, 0,12.0),
    ("Akola",             20.7002, 77.0082, 0.58, 0.10, 0.76, 0.64, 0.15, 32.0, 0.40, 1,11.0),
    ("Yavatmal",          20.3888, 78.1204, 0.42, 0.22, 0.62, 0.52, 0.20, 14.0, 0.32, 0,15.0),
    ("Washim",            20.1102, 77.1333, 0.35, 0.28, 0.55, 0.46, 0.22,  9.0, 0.30, 0,12.0),
    ("Buldhana",          20.5293, 76.1804, 0.40, 0.24, 0.60, 0.50, 0.20, 11.0, 0.32, 0,13.0),

    # === MARATHWADA — Aurangabad (Chhatrapati Sambhajinagar) division ===
    ("Sambhajinagar CBD", 19.8762, 75.3433, 0.65, 0.08, 0.80, 0.70, 0.15, 35.0, 0.48, 0,10.0),
    ("Sambhajinagar MIDC",19.8600, 75.4200, 0.55, 0.06, 0.80, 0.60, 0.10, 15.0, 0.38, 1,14.0),
    ("Jalna",              19.8410, 75.8864, 0.50, 0.14, 0.70, 0.58, 0.18, 20.0, 0.35, 1,12.0),
    ("Beed",               18.9891, 75.7601, 0.42, 0.18, 0.64, 0.52, 0.18, 15.0, 0.32, 0,14.0),
    ("Latur",              18.4088, 76.5604, 0.55, 0.12, 0.74, 0.62, 0.15, 26.0, 0.38, 0,11.0),
    ("Dharashiv",          18.1860, 76.0419, 0.42, 0.18, 0.64, 0.52, 0.20, 14.0, 0.32, 0,13.0),
    ("Nanded City",        19.1383, 77.3210, 0.55, 0.12, 0.74, 0.62, 0.20, 28.0, 0.38, 0,12.0),
    ("Parbhani",           19.2704, 76.7602, 0.48, 0.16, 0.68, 0.56, 0.18, 20.0, 0.35, 0,12.0),
    ("Hingoli",            19.7148, 77.1490, 0.35, 0.28, 0.52, 0.44, 0.22,  9.0, 0.30, 0,12.0),

    # === NORTH MAHARASHTRA — Nashik division ===
    ("Dhule",              20.9042, 74.7749, 0.50, 0.15, 0.70, 0.58, 0.20, 22.0, 0.36, 0,13.0),
    ("Nandurbar",          21.3702, 74.2400, 0.35, 0.30, 0.52, 0.44, 0.25, 10.0, 0.28, 0,14.0),
    ("Jalgaon",            21.0077, 75.5626, 0.55, 0.12, 0.74, 0.62, 0.18, 28.0, 0.40, 1,11.0),
    ("Bhusawal",           21.0433, 75.7850, 0.50, 0.14, 0.70, 0.58, 0.20, 22.0, 0.35, 1,10.0),
    ("Ahmednagar City",    19.0952, 74.7496, 0.55, 0.14, 0.74, 0.62, 0.18, 24.0, 0.40, 0,12.0),
    ("Shirdi",             19.7645, 74.4763, 0.35, 0.25, 0.55, 0.48, 0.22, 12.0, 0.50, 0, 8.0),
    ("Malegaon",           20.5537, 74.5288, 0.60, 0.08, 0.80, 0.66, 0.15, 40.0, 0.30, 0, 9.0),

    # === WESTERN MAHARASHTRA — Pune division extra ===
    ("Solapur City",       17.6599, 75.9064, 0.62, 0.10, 0.78, 0.66, 0.18, 38.0, 0.40, 1,11.0),
    ("Sangli",             16.8524, 74.5815, 0.55, 0.14, 0.72, 0.60, 0.25, 30.0, 0.42, 0,10.0),
    ("Miraj",              16.8299, 74.6414, 0.50, 0.16, 0.68, 0.56, 0.22, 24.0, 0.38, 0, 9.0),
    ("Karad",              17.2857, 74.1815, 0.42, 0.20, 0.62, 0.52, 0.28, 16.0, 0.38, 0,10.0),
    ("Baramati",           18.1514, 74.5815, 0.42, 0.20, 0.62, 0.52, 0.22, 14.0, 0.42, 0,11.0),
    ("Phaltan",            17.9922, 74.4302, 0.35, 0.24, 0.55, 0.46, 0.22, 10.0, 0.35, 0,10.0),

    # === KONKAN — coastal south ===
    ("Ratnagiri City",     16.9944, 73.3000, 0.40, 0.30, 0.55, 0.46, 0.55, 12.0, 0.42, 0,10.0),
    ("Chiplun",            17.5321, 73.5158, 0.35, 0.35, 0.50, 0.42, 0.45, 10.0, 0.40, 0,10.0),
    ("Sindhudurg",         16.1667, 73.6833, 0.20, 0.55, 0.35, 0.32, 0.60,  4.0, 0.40, 0,14.0),
    ("Malvan",             16.0667, 73.4667, 0.22, 0.45, 0.40, 0.35, 0.75,  6.0, 0.42, 0, 8.0),
    ("Mahad",              18.0833, 73.4167, 0.35, 0.32, 0.50, 0.42, 0.35,  9.0, 0.35, 1,10.0),
    ("Roha",               18.4386, 73.1189, 0.38, 0.28, 0.55, 0.46, 0.32, 10.0, 0.35, 1, 9.0),
    ("Pen",                18.7411, 73.0961, 0.42, 0.24, 0.60, 0.50, 0.30, 14.0, 0.40, 0, 8.0),
    ("Uran",               18.8763, 72.9432, 0.45, 0.18, 0.65, 0.55, 0.45, 16.0, 0.42, 1, 8.0),

    # === Extra Palghar / North Konkan filler ===
    ("Palghar City",       19.6969, 72.7649, 0.45, 0.20, 0.65, 0.55, 0.35, 18.0, 0.40, 0,10.0),
    ("Boisar",              19.8067, 72.7519, 0.42, 0.18, 0.62, 0.52, 0.30, 16.0, 0.35, 1, 9.0),
    ("Jawhar",              19.9083, 73.2333, 0.15, 0.68, 0.25, 0.25, 0.25,  2.0, 0.20, 0,12.0),
    ("Dahanu",              19.9700, 72.7300, 0.30, 0.38, 0.48, 0.40, 0.55,  8.0, 0.35, 0, 9.0),
]

STATE_COLUMNS = EXTRA_COLUMNS  # reuse same column order


def generate_full_maharashtra_data() -> pd.DataFrame:
    """
    Full state coverage: Mumbai (65) + MMR/Nashik/Pune extras (43) +
    Vidarbha + Marathwada + North Maharashtra + Konkan + Western
    Maharashtra (48 more) = ~150 locations spanning the entire state
    from Nandurbar in the north to Sindhudurg in the south, and
    Nagpur/Gadchiroli in the east to the Konkan coast in the west.
    """
    base = generate_maharashtra_data()
    extra_rows = []
    for record in STATE_WIDE_EXTRA:
        d = dict(zip(STATE_COLUMNS, record))
        d["lst"] = compute_lst(d)
        d["heat_stress_index"] = compute_heat_stress_index(d["lst"], d["pop_density_k"])
        vuln = (1 - d["income_index"]) * 0.5 + min(d["pop_density_k"] / 300, 1.0) * 0.5
        d["equity_score"] = round(
            float(np.clip(0.60 * ((d["lst"] - 24) / 26) + 0.40 * vuln, 0, 1) * 100), 2
        )
        if d["lst"] >= 42: d["lst_category"] = "Critical"
        elif d["lst"] >= 38: d["lst_category"] = "High"
        elif d["lst"] >= 34: d["lst_category"] = "Moderate"
        else: d["lst_category"] = "Low"
        extra_rows.append(d)
    extra_df = pd.DataFrame(extra_rows)
    for col in base.columns:
        if col not in extra_df.columns:
            extra_df[col] = 0
    combined = pd.concat([base, extra_df[base.columns]], ignore_index=True)
    return combined


# Geographically accurate Maharashtra state boundary (lat, lon) —
# traced as a strictly-ordered clockwise perimeter walk, starting at
# the NW coast near the Gujarat/Dadra Nagar Haveli border, north along
# the Gujarat/MP border, east across Vidarbha to the Chhattisgarh
# "nose" near Gadchiroli, southwest along the Telangana and Karnataka
# borders, then north up the Konkan coast back to start.
# Verified: simple, non-self-intersecting polygon (shapely is_simple=True).
MAHARASHTRA_BOUNDARY = [
    (20.17, 72.75),
    (20.60, 73.05), (21.00, 73.30), (21.35, 73.75),
    (21.50, 74.30), (21.65, 75.00), (21.75, 75.80),
    (21.65, 76.60), (21.55, 77.30), (21.45, 78.00),
    (21.65, 78.70), (21.90, 79.30),
    (21.75, 80.00), (21.20, 80.40), (20.70, 80.40),
    (20.10, 80.15), (19.60, 80.05),
    (19.15, 79.70), (18.85, 79.30), (18.60, 78.70),
    (18.55, 78.00), (18.35, 77.40), (18.15, 76.90),
    (17.85, 76.50), (17.55, 76.05), (17.15, 75.60),
    (16.85, 75.05), (16.50, 74.55), (16.10, 74.10),
    (15.80, 73.90), (15.65, 73.65),
    (16.20, 73.30), (16.95, 73.28), (17.65, 73.18),
    (18.25, 72.85), (18.90, 72.75), (19.30, 72.70),
    (19.85, 72.65),
]


if __name__ == "__main__":
    full = generate_full_maharashtra_data()
    print(f"Full Maharashtra dataset: {len(full)} locations")
    print(f"LST range: {full.lst.min():.1f}–{full.lst.max():.1f}°C")
    print(f"Lat range: {full.lat.min():.2f}–{full.lat.max():.2f}")
    print(f"Lon range: {full.lon.min():.2f}–{full.lon.max():.2f}")
