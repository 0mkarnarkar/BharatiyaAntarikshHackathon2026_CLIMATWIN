# CLIMATWIN 🌡️
### Causal Learning-Integrated Multi-modal AI Twin for Urban Heat Neutralization
**Physics-Informed AI Platform for Urban Heat Mitigation — Mumbai Pilot**

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone or unzip the project
cd CLIMATWIN

# 2. Install dependencies (all free, open-source)
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```
Open http://localhost:8501 in your browser. That's it — zero API keys, zero cost.

---

## 📁 Project Structure

```
CLIMATWIN/
├── app.py                    # Streamlit dashboard (5 tabs)
├── requirements.txt          # All dependencies
├── modules/
│   ├── data_generator.py     # Physics-informed Mumbai LST data (65 neighborhoods)
│   ├── ml_model.py           # Random Forest + SHAP driver analysis
│   ├── scenarios.py          # Cooling intervention simulator (5 types)
│   └── equity.py             # Thermal Equity Index calculator
└── README.md
```

---

## 🗺️ Dashboard Tabs

| Tab | What it shows |
|-----|--------------|
| 🗺️ Heat Stress Map | Interactive LST heatmap for all 65 Mumbai neighborhoods |
| 📊 Driver Analysis | SHAP-based global + per-neighborhood driver ranking |
| 🌿 Scenario Studio | Sliders to simulate 5 interventions, before/after map + cost |
| ⚖️ Equity Index | Vulnerability quadrant + equity score map |
| 🤖 Model Insights | Predicted vs observed, residuals, architecture roadmap |

---

## 🧠 AI Stack

| Component | Current (Prototype) | Production (Roadmap) |
|-----------|-------------------|---------------------|
| Heat Prediction | Random Forest | Physics-Informed Neural Network (DeepXDE) |
| Driver Analysis | SHAP | Causal DAG (DoWhy / pgmpy) |
| Heat Propagation | Tabular features | Temporal Graph Neural Network (PyG) |
| Intervention Opt. | Rule-based simulator | Multi-Agent RL (Ray RLlib) |
| Data Source | Physics-informed synthetic | Landsat 8 + ECOSTRESS (GEE) |

---

## 🌡️ Physics Formula

LST is generated using a surface energy balance proxy:
```
LST = T_base (31.5°C)
    + 8.20 × Building Density
    + 6.50 × Impervious Surface Fraction
    − 7.80 × Green Cover Fraction
    + 3.10 × Road Density
    − 4.50 × Water Proximity
    + 2.80 × Industrial Flag
    + N(0, 0.65)   ← sensor noise
```
**Range achieved:** 24.8°C (SGNP) → 47.4°C (Dharavi) — matches real Mumbai observations.

---

## 🌿 Cooling Interventions Simulated

| Intervention | Cooling Effect | Source |
|-------------|---------------|--------|
| Green Roofs | −0.09°C per 1% GCF increase | Santamouris 2014 |
| Street Trees | −0.14°C per 1% canopy increase | Bowler et al. 2010 |
| Cool Roofs | −2.30°C per 0.1 albedo increase | Akbari et al. 2009 |
| Water Features | −1.80°C per 0.1 water proximity | Völker et al. 2013 |
| Urban Forest | −0.22°C per 1% cover increase | Ziter et al. 2019 |

---

## ⚖️ Equity Index Formula

```
Equity Score = 0.40 × Heat Exposure
             + 0.35 × Vulnerability (income + density)
             + 0.15 × Green Cover Deficit
             − 0.10 × Adaptive Capacity
```
**Quadrants:** 🔴 Urgent (hot + vulnerable) | 🟠 Monitor | 🟡 Support | 🟢 Stable

---

## 🔜 Plugging in Real Data

To connect to real Landsat 8 data (free, needs GEE account):
```python
# In modules/data_generator.py, replace generate_data() with:
import ee
ee.Initialize()
mumbai = ee.Geometry.Rectangle([72.77, 18.89, 73.00, 19.27])
lst_image = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
             .filterBounds(mumbai)
             .filterDate('2024-04-01', '2024-06-30')
             .select(['ST_B10'])
             .mean()
             .multiply(0.00341802).add(149.0).subtract(273.15))  # K → °C
```

---

## 📊 Model Performance

- **R²:** ~0.97 (high due to physics-informed synthetic data)
- **MAE:** ~0.3°C
- **CV R²:** ~0.93 ± 0.04

---

## 💰 Cost to Run

**₹0** — 100% free tools:
- Data: Synthetic (real: Landsat 8/ECOSTRESS free from NASA/USGS)
- ML: scikit-learn, SHAP (open-source)
- Dashboard: Streamlit Community Cloud (free hosting)
- Maps: Plotly + OpenStreetMap tiles (no API key)

---

*Built for PS1 · Urban Heat Mitigation · National AI/ML Hackathon*
*Team CLIMATWIN — Winners of Kleos 4.0, RAIT ACM National Hackathon*
