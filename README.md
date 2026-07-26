<p align="center">
  <strong>CLIMATWIN</strong><br>
  <em>Causal Learning-Integrated Multi-modal AI Twin for Urban Heat Neutralization</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Cost-₹0_(Fully_Free)-0EA5E9?style=flat-square" alt="Cost">
</p>

---

**CLIMATWIN** is a physics-informed AI platform for urban heat mitigation, covering **154 zones** across Mumbai and Maharashtra. It combines machine learning, SHAP-based causal analysis, and interactive simulation to help urban planners identify heat hotspots, evaluate cooling interventions, and prioritize equity-driven action.

> Built for **PS-1: Urban Heat Mitigation** — Bharatiya Antariksh Hackathon 2026

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/0mkarnarkar/BharatiyaAntarikshHackathon2026_CLIMATWIN.git
cd BharatiyaAntarikshHackathon2026_CLIMATWIN

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser. No API keys or paid services required.

---

## Project Structure

```
CLIMATWIN/
├── app.py                        # Main Streamlit dashboard (10 modules)
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Theme and server configuration
├── modules/
│   ├── data_generator.py         # Physics-informed synthetic LST generator (154 zones)
│   ├── ml_model.py               # Random Forest model + SHAP explainability
│   ├── scenarios.py              # Cooling intervention simulator (5 strategies)
│   └── equity.py                 # Thermal Equity Index calculator
└── site/
    ├── index.html                # Static landing page
    ├── styles.css                # Landing page styles
    └── script.js                 # Landing page interactions
```

---

## Dashboard Modules

| Module | Description |
|--------|-------------|
| **Heat Map** | Interactive pixel-level LST heatmap across all 154 Maharashtra zones |
| **What-If Engine** | Counterfactual analysis — swap urban features between zones to predict LST changes |
| **Materials** | Surface material comparison with albedo, cooling potential, cost, and durability |
| **Analysis** | SHAP-based global and per-neighborhood driver ranking |
| **Predictions** | Predicted vs. observed LST, residual diagnostics, model architecture |
| **Optimization** | Multi-intervention portfolio simulator with cost-benefit analysis |
| **Equity** | Thermal Equity Index with vulnerability quadrant classification |
| **Alerts** | Heat stress alerts for critical zones |
| **Reports** | Exportable summary reports |
| **Settings** | Dashboard configuration and preferences |

---

## Technical Architecture

| Component | Current Implementation | Production Roadmap |
|-----------|----------------------|-------------------|
| Heat Prediction | Random Forest (scikit-learn) | Physics-Informed Neural Network (DeepXDE) |
| Driver Analysis | SHAP (TreeExplainer) | Causal DAG (DoWhy / pgmpy) |
| Heat Propagation | Tabular feature engineering | Temporal Graph Neural Network (PyG) |
| Intervention Optimization | Rule-based simulator | Multi-Agent RL (Ray RLlib) |
| Data Source | Physics-informed synthetic | Landsat 8 + ECOSTRESS via Google Earth Engine |

---

## Physics Model

Land Surface Temperature is generated using a surface energy balance proxy calibrated against observed Mumbai thermal profiles:

```
LST = T_base (31.5°C)
    + 8.20 × Building Density
    + 6.50 × Impervious Surface Fraction
    − 7.80 × Green Cover Fraction
    + 3.10 × Road Density
    − 4.50 × Water Proximity
    + 2.80 × Industrial Flag
    + N(0, 0.65)                          # sensor noise term
```

**Validated range:** 24.8°C (SGNP forest buffer) to 47.4°C (Dharavi), consistent with real-world observations from Landsat 8 thermal band studies over Mumbai.

---

## Cooling Interventions

All intervention coefficients are derived from peer-reviewed urban climate literature:

| Intervention | Cooling Effect | Reference |
|-------------|---------------|-----------|
| Green Roofs | −0.09°C per 1% GCF increase | Santamouris, 2014 |
| Street Trees | −0.14°C per 1% canopy increase | Bowler et al., 2010 |
| Cool Roofs | −2.30°C per 0.1 albedo increase | Akbari et al., 2009 |
| Water Features | −1.80°C per 0.1 water proximity | Völker et al., 2013 |
| Urban Forest | −0.22°C per 1% cover increase | Ziter et al., 2019 |

---

## Thermal Equity Index

The equity framework prioritizes zones facing compounded heat and social vulnerability:

```
Equity Score = 0.40 × Heat Exposure
             + 0.35 × Vulnerability (income + density)
             + 0.15 × Green Cover Deficit
             − 0.10 × Adaptive Capacity
```

**Classification quadrants:** Urgent (high heat + high vulnerability) | Monitor | Support | Stable

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² | ~0.97 |
| MAE | ~0.3°C |
| Cross-validated R² | 0.93 ± 0.04 |

> High R² is expected given physics-informed synthetic data with known ground truth. Real-world deployment with satellite data will require re-validation.

---

## Connecting Real Data

To transition from synthetic to real Landsat 8 thermal data (free via Google Earth Engine):

```python
# Replace generate_data() in modules/data_generator.py
import ee
ee.Initialize()

mumbai = ee.Geometry.Rectangle([72.77, 18.89, 73.00, 19.27])
lst_image = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
             .filterBounds(mumbai)
             .filterDate('2024-04-01', '2024-06-30')
             .select(['ST_B10'])
             .mean()
             .multiply(0.00341802).add(149.0).subtract(273.15))  # Kelvin to Celsius
```

---

## Cost

**Zero.** The entire stack runs on free, open-source tools:

| Layer | Technology |
|-------|-----------|
| Data | Synthetic (production: Landsat 8 / ECOSTRESS — free from NASA/USGS) |
| ML | scikit-learn, SHAP |
| Dashboard | Streamlit Community Cloud |
| Maps | Plotly, Folium, OpenStreetMap tiles |

---

## Team

**Team CLIMATWIN** — Bharatiya Antariksh Hackathon 2026, PS-1: Urban Heat Mitigation

---
