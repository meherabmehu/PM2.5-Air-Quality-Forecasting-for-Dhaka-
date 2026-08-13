# 🌍 24-Hour Ahead Daily-Average PM2.5 Forecasting for Dhaka, Bangladesh
### Research-Grade Causal Hybrid Ridge-Residual Boosting Architecture & Live Verification Interface

[![Live API Integration](https://img.shields.io/badge/Open--Meteo%20%26%20wttr.in-Live%20API-185FA5)](https://open-meteo.com/)
[![Model R2 Score](https://img.shields.io/badge/Test%20R%C2%B2-0.8650-2CA02C)](#empirical-performance--model-comparison)
[![WHO/EPA Compliance](https://img.shields.io/badge/AQI%20Severity-WHO%20%2F%20EPA-FF9800)](#operational-public-health-classification)
[![Python & Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-E91E63)](app.py)

---

## 📖 Executive Summary & Problem Statement

Air pollution is one of the most critical environmental and public health emergencies in Bangladesh. **Dhaka** is consistently ranked among the world's most polluted megacities. Fine particulate matter (**$\text{PM}_{2.5}$**), consisting of particles with an aerodynamic diameter less than 2.5 micrometers, penetrates deep into the alveolar regions of the human lungs and bloodstream, causing severe cardiovascular and respiratory diseases, premature mortality, and reduced life expectancy.

While existing air quality monitoring stations report *instantaneous* ambient pollution levels, they do not offer sufficient **advance warning** for municipal authorities, healthcare providers, educational institutions, or citizens to take preventive actions. Accurately forecasting $\text{PM}_{2.5}$ concentrations **24 hours in advance** enables timely public health advisories, proactive school/workload management, and targeted environmental interventions.

However, 24-hour ahead $\text{PM}_{2.5}$ forecasting remains a mathematically complex challenge due to:
* **Nonlinear Meteorological Dynamics:** The complex, dynamic relationships between particulate accumulation and meteorological variables such as temperature, relative humidity, wind speed, and precipitation (e.g., rainfall scavenging/washout and calm-wind atmospheric stagnation).
* **Temporal & Seasonal Baseline Shift:** Dhaka exhibits an upward seasonal pollution drift between training periods and future years. Standard statistical linear models fail to capture non-linear weather thresholds, while standalone tree-based boosting models make *step-function predictions* that cannot extrapolate linearly across new seasonal levels.

This research introduces an empirically verified **Hybrid Ridge-Residual Gradient Boosting Architecture** that solves both challenges, achieving state-of-the-art predictive performance (**Test $R^2 = 0.8650$, $\text{RMSE} = 21.66\ \mu\text{g/m}^3$**) on an unseen multi-year chronological test set.

---

## 🧠 Methodology & Hybrid Ridge-Residual Boosting Architecture

To overcome the extrapolation limits of standard tree boosting and the linearity constraints of regression, we engineered a two-stage hybrid framework:

$$\hat{y}_{\text{hybrid}}(X) = \hat{y}_{\text{Ridge}}(X) + \hat{y}_{\text{TreeRes}}(X)$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 HYBRID RIDGE-RESIDUAL BOOSTING PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ STAGE 1: Causal Linear Autoregressive Anchor (RidgeCV)                      │
│   ├── Learns continuous 24h momentum & autoregression (R² = 0.8533)         │
│   └── Extrapolates cleanly into future years without boundary clipping      │
├─────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: Non-Linear Meteorological Correction (HistGradientBoosting Tree)   │
│   ├── Fits gradient boosting trees purely on training residuals             │
│   │   (res_train = y_train - p_ridge_train)                                 │
│   └── Captures rainfall washout, calm-wind stagnation, and diurnal cycles   │
├─────────────────────────────────────────────────────────────────────────────┤
│ FINAL OUTPUT: 24h-Ahead Daily-Average Forecast (Test R² = 0.8650)           │
│   └── Mapped to official WHO / EPA AQI Public Health Severity Bands         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Technical Contributions:
1. **Core 92 Causal Feature Set:** We construct unshifted historical EMAs across multiple spans (`3h, 6h, 12h, 24h, 48h, 72h, 168h`), causal lag vectors up to 1 week, 24-hour rolling momentum differences, trend acceleration, and cumulative precipitation washout features (`rainfall_cum_24h/48h/72h`).
2. **Zero Data Leakage:** Strictly chronological `70 / 15 / 15` split (`2016–2020` train, `2020–2021` validation, `2021–2022` test), with all feature transformers and imputers fitted solely on the training distribution.
3. **Automated Live Data Assimilation:** The live dashboard shows **human websites**, not JSON, as proof. Current PM2.5 comes from the [weather.com Dhaka Air Quality](https://weather.com/forecast/air-quality/l/23.81,90.41) page (µg/m³ line — the big number on that page is AQI, not PM2.5). Temperature, humidity, wind and rainfall come from the official Google Weather API when a key is provided, verified against the [Google Weather card for Dhaka](https://www.google.com/search?q=weather+in+dhaka&hl=en). Without a Google key the four weather fields fall back to [weather.com Today](https://weather.com/weather/today/l/23.81,90.41?unit=m). Hourly history for lags is still Open-Meteo, clipped to the current Dhaka hour. Timestamps are Bangladesh Standard Time (BST / UTC+6).

---

## 📊 Empirical Performance & Model Comparison

Tested on the unseen chronological test set (`y_test`, 7,754 hourly records from `2021–2022`), our Hybrid Ridge-Residual models outperform every standalone direct tree benchmark:

| Model / Architecture | Test $R^2$ Score | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | Status / Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **`HistGB_Hybrid_Res` (Ridge + Residual Tree)** | **0.8650** | **21.66** | **15.69** | **Champion Model 🏆** |
| **`Hybrid_Champion_Ensemble` (Top-4 Hybrid Blend)** | **0.8642** | **21.73** | **15.75** | **Ensemble Leader 🥇** |
| **`LightGBM_Hybrid_Res` (Ridge + Residual LGBM)** | **0.8631** | **21.82** | **15.81** | **Hybrid Champion 🥈** |
| **`XGBoost_Hybrid_Res` (Ridge + Residual XGB)** | **0.8624** | **21.87** | **15.86** | **Hybrid Champion 🥉** |
| **`CatBoost_Hybrid_Res` (Ridge + Residual CB)** | **0.8618** | **21.92** | **15.89** | **Hybrid Champion** |
| `XGBoost_RMSE_Test` (Direct Tree Benchmark) | 0.8558 | 22.40 | 16.06 | Direct Tree Benchmark |
| `LightGBM_RMSE_Test` (Direct Tree Benchmark) | 0.8558 | 22.40 | 16.11 | Direct Tree Benchmark |
| `RidgeCV_Linear_Test` (Scaled Linear Anchor) | 0.8533 | 22.63 | 16.02 | Autoregressive Baseline |
| `CatBoost_RMSE_Test` (Direct Tree Benchmark) | 0.8495 | 22.89 | 16.36 | Direct Tree Benchmark |

---

## 🚀 Quick-Start Instructions (From `git clone` to Live App)

You can launch the **Real-Time Live Testing & Verification Web Dashboard** on any computer (Windows, macOS, or Linux) in under 60 seconds.

### Step 1: Clone the Repository
Open your terminal (or PowerShell on Windows) and clone the project:
```bash
# Step 1: Navigate to your preferred directory (e.g., E:\ drive)
cd E:\

# Step 2: Clone the GitHub repository
git clone https://github.com/meherabmehu/PM2.5-Air-Quality-Forecasting-for-Dhaka-.git
# Step 3: Enter the repository folder using the full path
cd E:\PM2.5-Air-Quality-Forecasting-for-Dhaka-
```

### Step 2: Install Required Dependencies
Install the required Python packages (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`):
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Real-Time Live Dashboard
Run the Streamlit interactive application:
```bash
streamlit run app.py
```

* **What happens next:**
  * Your web browser will automatically open **`http://localhost:8501`**.
  * Click **`🔄 Fetch Live Dhaka Data & Predict Now`** to pull real-time weather (`wttr.in/Dhaka`) and Open-Meteo air quality in **Dhaka Local Time (BST)**, run live inference, and display the 24-hour ahead forecast!

---

## 📓 Jupyter & Kaggle Research Notebooks

The repository includes two research notebooks located in the `Notebook/` directory:

1. **`Notebook/Live_Dhaka_PM25_Forecaster.ipynb` (1-Click Live Verification Notebook):**
   * Can be executed in Kaggle, Google Colab, or Jupyter Notebook.
   * Connects to Open-Meteo and `wttr.in`, generates the Core 92 Causal Feature Set, runs the Champion Hybrid Model, outputs WHO/EPA AQI severity bands, and saves a publication-ready figure (`dhaka_live_forecast.png`).
2. **`Notebook/Notebook_R2_095_Kaggle_Ready.ipynb` (Complete 13-Model Benchmark Notebook):**
   * Trains and evaluates all 13 individual models across 6 distinct AI/ML paradigms, performs Scipy SLSQP convex optimization, and exports full diagnostic tables and charts.

---

## 📁 Repository Structure

```
PM2.5-Air-Quality-Forecasting-for-Dhaka-/
├── README.md                                  # Research documentation & Quick-Start guide
├── app.py                                     # Streamlit Real-Time Live Web Dashboard app
├── requirements.txt                           # Python dependencies for deployment
├── train_live_model.py                        # Standalone script to re-fit Champion model
├── models/
│   ├── live_hybrid_champion.pkl               # Pre-trained Champion model (loads instantly)
│   └── live_feature_names.json                # Feature schema & R² = 0.8650 test metrics
├── Notebook/
│   ├── Live_Dhaka_PM25_Forecaster.ipynb       # Live real-time Jupyter/Kaggle testing notebook
│   └── Notebook_R2_095_Kaggle_Ready.ipynb     # Complete 13-model thesis benchmark notebook
├── Dataset/
│   ├── final_dataset_clean.csv                # Clean 6-column meteorological dataset
│   └── final_dataset_feature_engineered.csv   # Complete multivariate dataset
├── Output/                                    # Exported thesis charts, tables & model logs
└── Report Summery/                            # Academic thesis progress reports & proposal
```

---

## 📜 BibTeX & Academic Citation

If you use this dataset, methodology, or live verification interface in your research, please cite:

```bibtex
@article{DhakaPM25Forecast2026,
  title={Multivariate 24-Hour Ahead Daily-Average PM2.5 Forecasting in Dhaka Using Hybrid Ridge-Residual Gradient Boosting},
  author={Talukder, Meherab Hossain and Collaborators},
  journal={Journal of Environmental Management / Atmospheric Environment},
  year={2026},
  note={GitHub Repository and Live Verification Suite: https://github.com/meherabmehu/PM2.5-Air-Quality-Forecasting-for-Dhaka-}
}
```
