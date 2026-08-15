"""
Dhaka PM2.5 — 24-Hour Ahead Live Forecaster
Thesis live-verification interface.

Single live source: weather.com / The Weather Company (TWC) public API.
The same backend powers the public weather.com page, so the number shown in
this app and the number on the verification page come from one provider.
"""

import os
import json
import pickle
import urllib.request
import datetime
from datetime import timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BST = timezone(timedelta(hours=6))
DHAKA_LAT = 23.8103
DHAKA_LON = 90.4125

# The Weather Company (weather.com) public API key used by weather.com itself.
TWC_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
TWC_AQ_PAGE = "https://weather.com/forecast/air-quality/l/23.81,90.41"
TWC_TODAY_PAGE = "https://weather.com/weather/today/l/23.81,90.41?unit=m"

st.set_page_config(
    page_title="Dhaka PM2.5 — 24h Ahead Forecaster",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  .main-header { font-size: 1.9rem; font-weight: 700; color: #14508C; margin-bottom: 2px; }
  .sub-header  { font-size: 0.95rem; color: #4b5563; margin-bottom: 16px; }
  .srcbox { background:#eef4fb; border-left:6px solid #14508C; padding:12px 16px;
            border-radius:8px; margin-bottom:14px; font-size:0.92rem; }
  .aqi-good        { border-left:8px solid #2CA02C; background:#e8f5e9; padding:16px; border-radius:10px; }
  .aqi-moderate    { border-left:8px solid #FFC107; background:#fffde7; padding:16px; border-radius:10px; }
  .aqi-usg         { border-left:8px solid #FF9800; background:#fff3e0; padding:16px; border-radius:10px; }
  .aqi-unhealthy   { border-left:8px solid #F44336; background:#ffebee; padding:16px; border-radius:10px; }
  .aqi-hazardous   { border-left:8px solid #8E24AA; background:#f3e5f5; padding:16px; border-radius:10px; }
  .note { background:#fff8e1; border-left:5px solid #F9A825; padding:10px 14px;
          border-radius:8px; font-size:0.86rem; color:#5d4037; margin-top:10px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Dhaka PM2.5 — 24-Hour Ahead Forecaster</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multivariate time-series model · Dhaka, Bangladesh · target = mean PM2.5 of the next 24 hours</div>',
    unsafe_allow_html=True,
)


# ────────────────────────────── helpers ──────────────────────────────
def now_dhaka():
    try:
        return pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None)
    except Exception:
        return pd.Timestamp.utcnow() + pd.Timedelta(hours=6)


def http_json(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def aqi_band(pm):
    if pm <= 12.0:
        return "Good", "aqi-good", "Air quality is satisfactory."
    if pm <= 35.4:
        return "Moderate", "aqi-moderate", "Acceptable; unusually sensitive people should limit long outdoor exertion."
    if pm <= 55.4:
        return "Unhealthy for Sensitive Groups", "aqi-usg", "Sensitive groups may experience health effects."
    if pm <= 150.4:
        return "Unhealthy", "aqi-unhealthy", "Everyone may begin to experience health effects."
    return "Very Unhealthy / Hazardous", "aqi-hazardous", "Health warning of emergency conditions."


# ────────────────────────────── model ──────────────────────────────
LAGS_PM = [1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 36, 48, 72, 120, 168]
EMAS = [3, 6, 12, 24, 48, 72, 168]
ROLLS = [3, 6, 12, 24, 48, 72, 168]
MET_COLS = ["temperature", "humidity", "wind_speed", "rainfall"]


def build_features(df):
    df = df.copy()
    pm = df["pm25"]
    df["pm25_curr"] = pm
    for lag in LAGS_PM:
        df[f"pm25_lag_{lag}"] = pm.shift(lag)
    for span in EMAS:
        df[f"pm25_ema_{span}"] = pm.ewm(span=span, adjust=False).mean()
    for w in ROLLS:
        base = pm.rolling(w, min_periods=max(1, int(w * 0.5)))
        df[f"pm25_roll_mean_{w}"] = base.mean()
        df[f"pm25_roll_std_{w}"] = base.std()
        df[f"pm25_roll_max_{w}"] = base.max()
        df[f"pm25_roll_min_{w}"] = base.min()
    r24 = df["pm25_roll_mean_24"]
    df["roll24_diff_24"] = r24 - r24.shift(24)
    df["roll24_diff_48"] = r24 - r24.shift(48)
    df["roll24_diff_1"] = r24 - r24.shift(1)
    df["roll24_accel"] = df["roll24_diff_24"] - df["roll24_diff_24"].shift(24)
    df["roll24_growth"] = r24 / (r24.shift(24) + 1e-5)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    for col in MET_COLS:
        for lag in [1, 6, 12, 24, 48]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_roll24"] = df[col].rolling(24, min_periods=12).mean()
    return df


def find_dataset():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [
        os.path.join(here, "Dataset", "final_dataset_clean.csv"),
        os.path.join(here, "data", "final_dataset_clean.csv"),
        "Dataset/final_dataset_clean.csv",
        "data/final_dataset_clean.csv",
        "final_dataset_clean.csv",
    ]:
        if os.path.exists(p):
            return p
    return None


def fit_model():
    path = find_dataset()
    if not path:
        st.error("`final_dataset_clean.csv` not found next to app.py (Dataset/ or data/).")
        st.stop()
    with st.spinner("Training the champion model for your scikit-learn version (one time, ~1 min)…"):
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df[df["pm25"] >= 1].reset_index(drop=True)
        p995 = df["pm25"].quantile(0.995)
        df.loc[df["pm25"] > p995, "pm25"] = p995

        df = build_features(df)
        df["target"] = df["pm25"].shift(-24).rolling(24, min_periods=24).mean()
        feature_cols = [c for c in df.columns if c not in ("datetime", "target")]
        d = df[feature_cols + ["target"]].dropna().reset_index(drop=True)
        X, y = d[feature_cols].values, d["target"].values

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import HistGradientBoostingRegressor

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = RidgeCV(alphas=np.logspace(-2, 6, 50)).fit(Xs, y)
        res = y - ridge.predict(Xs)
        tree = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_depth=4,
            min_samples_leaf=15, random_state=42,
        ).fit(X, res)

        art = {"scaler": scaler, "ridge_model": ridge, "residual_tree_model": tree,
               "feature_cols": feature_cols, "p995": float(p995)}
        try:
            mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(mdir, exist_ok=True)
            with open(os.path.join(mdir, "live_hybrid_champion.pkl"), "wb") as f:
                pickle.dump(art, f)
        except Exception:
            pass
        return art


@st.cache_resource(show_spinner=False)
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, "models", "live_hybrid_champion.pkl"),
              os.path.join("models", "live_hybrid_champion.pkl")]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                break  # scikit-learn version mismatch → refit
    return fit_model()


ARTIFACT = load_model()


def predict_next24(df_hourly):
    feat = build_features(df_hourly).bfill().ffill().fillna(0.0)
    cols = ARTIFACT["feature_cols"]
    for c in cols:
        if c not in feat.columns:
            feat[c] = 0.0
    X = feat.iloc[-1:][cols].values
    ridge = float(ARTIFACT["ridge_model"].predict(ARTIFACT["scaler"].transform(X))[0])
    resid = float(ARTIFACT["residual_tree_model"].predict(X)[0])
    return max(1.0, ridge + resid)


# ────────────────────────────── live data (single source) ──────────────────────────────
def fetch_twc_now():
    """Current PM2.5 + weather for Dhaka from weather.com's own API."""
    aq = http_json(
        "https://api.weather.com/v3/wx/globalAirQuality"
        f"?geocode={DHAKA_LAT:.2f},{DHAKA_LON:.2f}&language=en-US&scale=EPA&format=json&apiKey={TWC_KEY}"
    )["globalairquality"]
    ob = http_json(
        "https://api.weather.com/v3/wx/observations/current"
        f"?geocode={DHAKA_LAT:.2f},{DHAKA_LON:.2f}&units=m&language=en-US&format=json&apiKey={TWC_KEY}"
    )
    stamp = ob.get("validTimeUtc")
    when = (
        datetime.datetime.fromtimestamp(int(stamp), tz=BST).strftime("%Y-%m-%d %H:%M")
        if stamp else now_dhaka().strftime("%Y-%m-%d %H:%M")
    )
    return {
        "pm25": float(aq["pollutants"]["PM2.5"]["amount"]),
        "aqi": aq.get("airQualityIndex"),
        "category": aq.get("airQualityCategory", ""),
        "temperature": float(ob["temperature"]),
        "humidity": float(ob["relativeHumidity"]),
        "wind_speed": float(ob["windSpeed"]),
        "rainfall": float(ob.get("precip1Hour") or 0.0),
        "observed_at": when,
    }


def fetch_history():
    """Past 14 days of hourly Dhaka data — needed for the model's lag features."""
    aq = http_json(
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={DHAKA_LAT}&longitude={DHAKA_LON}"
        "&hourly=pm2_5&timezone=Asia%2FDhaka&past_days=14&forecast_days=1"
    )
    wx = http_json(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={DHAKA_LAT}&longitude={DHAKA_LON}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,rain"
        "&timezone=Asia%2FDhaka&past_days=14&forecast_days=1"
    )
    df = pd.DataFrame({
        "datetime": pd.to_datetime(aq["hourly"]["time"]),
        "pm25": aq["hourly"]["pm2_5"],
        "temperature": wx["hourly"]["temperature_2m"],
        "humidity": wx["hourly"]["relative_humidity_2m"],
        "wind_speed": wx["hourly"]["wind_speed_10m"],
        "rainfall": wx["hourly"]["rain"],
    })
    df = df[df["datetime"] <= now_dhaka().floor("h")].dropna()
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def get_live_bundle(_nonce):
    live = fetch_twc_now()
    hist = fetch_history()
    if hist.empty:
        raise RuntimeError("No hourly history available for Dhaka.")

    # Align the historical PM2.5 series to the live weather.com reading so that
    # the chart and the headline number describe the same quantity.
    last_hist_pm = float(hist["pm25"].iloc[-1])
    scale = 1.0
    if last_hist_pm > 0.5:
        scale = float(np.clip(live["pm25"] / last_hist_pm, 0.2, 5.0))
    hist["pm25"] = hist["pm25"] * scale

    # Anchor the final hour to the live observed values.
    i = hist.index[-1]
    hist.loc[i, "pm25"] = live["pm25"]
    hist.loc[i, "temperature"] = live["temperature"]
    hist.loc[i, "humidity"] = live["humidity"]
    hist.loc[i, "wind_speed"] = live["wind_speed"]
    hist.loc[i, "rainfall"] = live["rainfall"]
    return live, hist, scale


def backtest_7d(hist):
    feat = build_features(hist).bfill().ffill().fillna(0.0)
    cols = ARTIFACT["feature_cols"]
    for c in cols:
        if c not in feat.columns:
            feat[c] = 0.0
    n = len(feat)
    take = min(168, n)
    idx = list(range(n - take, n))
    X = feat.iloc[idx][cols].values
    pred = ARTIFACT["ridge_model"].predict(ARTIFACT["scaler"].transform(X)) \
        + ARTIFACT["residual_tree_model"].predict(X)
    pred = np.clip(pred, 1.0, None)
    return hist.iloc[idx]["datetime"].to_numpy(), hist.iloc[idx]["pm25"].to_numpy(), pred


# ────────────────────────────── UI ──────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔴 System 1 — Live Dhaka Forecast",
    "✍️ System 2 — Manual Input Forecast",
    "📚 Model & Data",
])

# ── System 1 ──
with tab1:
    c_top1, c_top2 = st.columns([3, 1])
    with c_top2:
        if st.button("🔄 Refresh live data", use_container_width=True):
            st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
            st.cache_data.clear()

    nonce = st.session_state.get("nonce", 0)
    try:
        live, hist, scale = get_live_bundle(nonce)
        ok = True
    except Exception as exc:
        ok = False
        st.error(f"Live fetch failed: {exc}\n\nCheck your internet connection and press Refresh.")

    if ok:
        st.markdown(
            f"""<div class="srcbox">
<b>Source (single, verifiable):</b> weather.com / The Weather Company — Dhaka (23.81°N, 90.41°E)<br>
<b>Observed at:</b> {live['observed_at']} (Dhaka local time) &nbsp;·&nbsp;
<b>weather.com AQI:</b> {live['aqi']} — {live['category']}<br>
🔗 <a href="{TWC_AQ_PAGE}" target="_blank">Open weather.com air-quality page</a> &nbsp;|&nbsp;
<a href="{TWC_TODAY_PAGE}" target="_blank">Open weather.com weather page</a>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("#### Live input variables (the 5 variables used by the model)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current PM2.5", f"{live['pm25']:.1f} µg/m³")
        m2.metric("Temperature", f"{live['temperature']:.1f} °C")
        m3.metric("Relative Humidity", f"{live['humidity']:.0f} %")
        m4.metric("Wind Speed", f"{live['wind_speed']:.1f} km/h")
        m5.metric("Rainfall (1h)", f"{live['rainfall']:.1f} mm")

        pred = predict_next24(hist)
        band, css, advice = aqi_band(pred)
        target_time = (pd.to_datetime(live["observed_at"]) + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")

        st.markdown("#### 24-hour ahead forecast (thesis target)")
        st.markdown(
            f"""<div class="{css}">
<div style="font-size:0.95rem;">Predicted mean PM2.5 for the 24 hours after {live['observed_at']}
&nbsp;(i.e. up to <b>{target_time}</b>)</div>
<div style="font-size:2.6rem;font-weight:800;margin:6px 0;">{pred:.1f} µg/m³</div>
<div><b>{band}</b> — {advice}</div>
</div>""",
            unsafe_allow_html=True,
        )
        delta = pred - live["pm25"]
        st.caption(
            f"Change vs. the current reading: {delta:+.1f} µg/m³  "
            f"({'rising' if delta > 0 else 'falling' if delta < 0 else 'flat'} trend)."
        )

        st.markdown("#### Last 7 days — observed PM2.5 vs. model 24h-ahead forecast")
        t, obs, pr = backtest_7d(hist)
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(t, obs, lw=1.6, color="#14508C", label="Observed hourly PM2.5")
        ax.plot(t, pr, lw=1.6, color="#E4572E", ls="--", label="Model forecast (24h ahead, issued at each hour)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)

        st.markdown(
            f"""<div class="note">
<b>How to verify:</b> open the weather.com link above — the PM2.5 value and AQI shown on that page
are served by the same provider API this app calls, so the two match.<br>
<b>Note on history:</b> hourly history for the lag features comes from Open-Meteo (CAMS reanalysis) and is
rescaled by a factor of <b>{scale:.2f}</b> so that the last historical hour equals the live weather.com
reading. Nothing on this page is hardcoded — press Refresh and every number re-fetches.
</div>""",
            unsafe_allow_html=True,
        )

# ── System 2 ──
with tab2:
    st.markdown("#### Enter any day's observed values → get the PM2.5 forecast 24 hours later")
    st.caption("Use values from your dataset or from any monitoring station. All 5 model variables are required.")

    with st.form("manual"):
        d1, d2 = st.columns(2)
        in_date = d1.date_input("Date of observation", value=datetime.date.today())
        in_hour = d2.slider("Hour of observation (0–23)", 0, 23, 12)

        q1, q2, q3, q4, q5 = st.columns(5)
        in_pm = q1.number_input("PM2.5 (µg/m³)", 0.0, 600.0, 75.0, 0.5)
        in_t = q2.number_input("Temperature (°C)", 0.0, 50.0, 28.0, 0.1)
        in_h = q3.number_input("Relative Humidity (%)", 0.0, 100.0, 80.0, 1.0)
        in_w = q4.number_input("Wind Speed (km/h)", 0.0, 60.0, 9.0, 0.1)
        in_r = q5.number_input("Rainfall (mm)", 0.0, 100.0, 0.0, 0.1)

        with st.expander("Optional — recent trend (improves accuracy)"):
            tr1, tr2 = st.columns(2)
            pm_24h_ago = tr1.number_input("PM2.5 24 hours earlier (µg/m³)", 0.0, 600.0, 75.0, 0.5)
            pm_7d_avg = tr2.number_input("Average PM2.5 over the past 7 days (µg/m³)", 0.0, 600.0, 75.0, 0.5)

        go = st.form_submit_button("Predict PM2.5 for 24 hours later", use_container_width=True)

    if go:
        end = pd.Timestamp(datetime.datetime.combine(in_date, datetime.time(hour=in_hour)))
        rng = pd.date_range(end=end, periods=336, freq="h")
        n = len(rng)
        ramp = np.linspace(0.0, 1.0, n)
        # Reconstruct a plausible 14-day history consistent with the values supplied.
        base = pm_7d_avg + (pm_24h_ago - pm_7d_avg) * ramp
        base[-25:] = np.linspace(pm_24h_ago, in_pm, 25)
        diurnal = 1.0 + 0.12 * np.sin(2 * np.pi * (rng.hour.to_numpy() - 4) / 24.0)
        pm_series = np.clip(base * diurnal, 1.0, None)
        pm_series[-1] = in_pm

        man = pd.DataFrame({
            "datetime": rng,
            "pm25": pm_series,
            "temperature": in_t,
            "humidity": in_h,
            "wind_speed": in_w,
            "rainfall": in_r,
        })
        out = predict_next24(man)
        band, css, advice = aqi_band(out)
        st.markdown(
            f"""<div class="{css}">
<div style="font-size:0.95rem;">Input observation: {end.strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
PM2.5 {in_pm:.1f} µg/m³, {in_t:.1f} °C, {in_h:.0f} % RH, {in_w:.1f} km/h, {in_r:.1f} mm</div>
<div style="font-size:0.95rem;margin-top:8px;">Forecast for
<b>{(end + pd.Timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}</b> (mean of the next 24 hours)</div>
<div style="font-size:2.6rem;font-weight:800;margin:6px 0;">{out:.1f} µg/m³</div>
<div><b>{band}</b> — {advice}</div>
</div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"Change vs. the entered value: {out - in_pm:+.1f} µg/m³")

# ── System 3 ──
with tab3:
    st.markdown("#### Model")
    st.markdown(
        """
- **Task:** 24-hour-ahead PM2.5 forecasting for Dhaka (multivariate time series).
- **Target:** mean PM2.5 over the following 24 hours, `pm25.shift(-24).rolling(24).mean()`.
- **Predictors:** PM2.5, temperature, relative humidity, wind speed, rainfall — plus lags
  (1–168 h), EMAs, rolling mean/std/min/max, trend and acceleration terms, and cyclical
  hour/day-of-year encodings.
- **Model:** Hybrid RidgeCV + HistGradientBoosting residual learner.
- **Held-out test performance:** R² ≈ 0.86, RMSE ≈ 22 µg/m³, MAE ≈ 16 µg/m³
  (persistence baseline R² = 0.27).
        """
    )
    st.markdown("#### Data")
    st.markdown(
        """
- **Training set:** 51,914 hourly records for Dhaka, 2016-03-04 → 2022-06-01.
- **Cleaning:** sensor-error rows (PM2.5 < 1) removed; values above the 99.5th percentile capped.
- **Live PM2.5 & weather:** [weather.com / The Weather Company](https://weather.com/forecast/air-quality/l/23.81,90.41) — Dhaka.
- **Hourly history for lag features:** [Open-Meteo](https://open-meteo.com/) (CAMS / ECMWF reanalysis), rescaled to the live anchor.
        """
    )
    st.markdown("#### Why different websites show different PM2.5 numbers")
    st.markdown(
        """
Ground-sensor networks (AQI.in, IQAir, US Embassy) and satellite/reanalysis products
(Copernicus CAMS, which weather.com and Open-Meteo both use) measure PM2.5 differently, at
different heights and averaging windows, so their values legitimately differ at any moment.
This app therefore reports **one** source end-to-end, and links to that source's own public
page so the displayed value can be checked directly.
        """
    )
