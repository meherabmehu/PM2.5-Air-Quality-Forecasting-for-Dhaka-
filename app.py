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
TWC_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
WEATHERCOM_AQ_PAGE = "https://weather.com/forecast/air-quality/l/23.81,90.41"
WEATHERCOM_TODAY_PAGE = "https://weather.com/weather/today/l/23.81,90.41?unit=m"
GOOGLE_WEATHER_PAGE = "https://www.google.com/search?q=weather+in+dhaka&hl=en"
GOOGLE_DEMO_KEY_PAGE = "https://developers.google.com/maps/documentation/weather/demo-key"
LIVE_REFRESH = timedelta(minutes=2)

st.set_page_config(
    page_title="Dhaka PM2.5 — 24h-Ahead Live Forecaster",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 1.85rem; font-weight: 700; color: #185FA5; margin-bottom: 0px; }
    .sub-header { font-size: 0.98rem; color: #495057; margin-top: 4px; margin-bottom: 14px; }
    .aqi-good { border-left: 8px solid #2CA02C; background: #e8f5e9; padding: 14px; border-radius: 8px; }
    .aqi-moderate { border-left: 8px solid #FFC107; background: #fffde7; padding: 14px; border-radius: 8px; }
    .aqi-unhealthy-sg { border-left: 8px solid #FF9800; background: #fff3e0; padding: 14px; border-radius: 8px; }
    .aqi-unhealthy { border-left: 8px solid #F44336; background: #ffebee; padding: 14px; border-radius: 8px; }
    .aqi-hazardous { border-left: 8px solid #8E24AA; background: #f3e5f5; padding: 14px; border-radius: 8px; }
    .source-box { background: #e3f2fd; padding: 10px 14px; border-radius: 8px; border-left: 6px solid #185FA5; margin-bottom: 12px; }
    .note-box { background: #fff8e1; padding: 10px 12px; border-radius: 8px; border-left: 5px solid #F9A825; margin: 8px 0 14px 0; }
    a { color: #185FA5; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Dhaka PM2.5 — 24-Hour Ahead Forecaster</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Live demo · Dhaka only · next-24h mean PM2.5</div>',
    unsafe_allow_html=True,
)


def now_dhaka_naive():
    try:
        return pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None)
    except Exception:
        return pd.Timestamp.utcnow() + pd.Timedelta(hours=6)


def http_json(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DhakaPM25Forecaster/1.0 (thesis live test)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fit_and_save_champion_model(base_dir):
    with st.spinner("Fitting local model for this scikit-learn version..."):
        data_candidates = [
            os.path.join(base_dir, "data", "final_dataset_clean.csv"),
            os.path.join(base_dir, "Dataset", "final_dataset_clean.csv"),
            "data/final_dataset_clean.csv",
            "Dataset/final_dataset_clean.csv",
            "final_dataset_clean.csv",
        ]
        data_path = next((p for p in data_candidates if os.path.exists(p)), None)
        if not data_path:
            st.error("Could not locate `final_dataset_clean.csv`.")
            return None

        df = pd.read_csv(data_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df[df["pm25"] >= 1].reset_index(drop=True)
        p995 = df["pm25"].quantile(0.995)
        df.loc[df["pm25"] > p995, "pm25"] = p995

        pm25 = df["pm25"]
        df["pm25_curr"] = pm25
        for lag in [1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 36, 48, 72, 120, 168]:
            df[f"pm25_lag_{lag}"] = pm25.shift(lag)
        for span in [3, 6, 12, 24, 48, 72, 168]:
            df[f"pm25_ema_{span}"] = pm25.ewm(span=span, adjust=False).mean()
        for w in [3, 6, 12, 24, 48, 72, 168]:
            base = pm25.rolling(w, min_periods=max(1, int(w * 0.5)))
            df[f"pm25_roll_mean_{w}"] = base.mean()
            df[f"pm25_roll_std_{w}"] = base.std()
            df[f"pm25_roll_max_{w}"] = base.max()
            df[f"pm25_roll_min_{w}"] = base.min()

        df["target"] = pm25.shift(-24).rolling(24, min_periods=24).mean()
        df["roll24_diff_24"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(24)
        df["roll24_diff_48"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(48)
        df["roll24_diff_1"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(1)
        df["roll24_accel"] = df["roll24_diff_24"] - df["roll24_diff_24"].shift(24)
        df["roll24_growth"] = df["pm25_roll_mean_24"] / (df["pm25_roll_mean_24"].shift(24) + 1e-5)

        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month
        df["doy"] = df["datetime"].dt.dayofyear
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

        for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
            for lag in [1, 6, 12, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            df[f"{col}_roll24"] = df[col].rolling(24, min_periods=12).mean()

        drop_always = ["datetime", "target"]
        feature_cols = [c for c in df.columns if c not in drop_always]
        df_sel = df[feature_cols + ["target"]].dropna().reset_index(drop=True)
        X_all = df_sel[feature_cols].values
        y_all = df_sel["target"].values

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import HistGradientBoostingRegressor

        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        m_ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
        m_ridge.fit(X_all_s, y_all)
        res_all = y_all - m_ridge.predict(X_all_s)
        m_res = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=15,
            random_state=42,
        )
        m_res.fit(X_all, res_all)

        artifact = {
            "scaler": scaler,
            "ridge_model": m_ridge,
            "residual_tree_model": m_res,
            "feature_cols": feature_cols,
            "p995": p995,
        }
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        try:
            with open(os.path.join(models_dir, "live_hybrid_champion.pkl"), "wb") as f:
                pickle.dump(artifact, f)
        except Exception:
            pass
        return artifact


@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "models", "live_hybrid_champion.pkl"),
        os.path.join("models", "live_hybrid_champion.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                st.warning("Pickle version mismatch — re-fitting a local model copy...")
                break
    return fit_and_save_champion_model(base_dir)


artifact = load_model_artifacts()


def get_aqi_band_info(pm25_val):
    if pm25_val <= 50:
        return "Good (0–50)", "aqi-good", "Satisfactory for most people."
    if pm25_val <= 100:
        return "Moderate (51–100)", "aqi-moderate", "Sensitive people should limit long outdoor exposure."
    if pm25_val <= 150:
        return "Unhealthy for Sensitive Groups (101–150)", "aqi-unhealthy-sg", "Sensitive groups may feel health effects."
    if pm25_val <= 200:
        return "Unhealthy (151–200)", "aqi-unhealthy", "Everyone may begin to feel health effects."
    return "Very Unhealthy / Hazardous (200+)", "aqi-hazardous", "Health warning of emergency conditions."


def engineer_live_features(df_input, feature_cols):
    df = df_input.copy()
    pm25 = df["pm25"]
    df["pm25_curr"] = pm25
    for lag in [1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 36, 48, 72, 120, 168]:
        df[f"pm25_lag_{lag}"] = pm25.shift(lag)
    for span in [3, 6, 12, 24, 48, 72, 168]:
        df[f"pm25_ema_{span}"] = pm25.ewm(span=span, adjust=False).mean()
    for w in [3, 6, 12, 24, 48, 72, 168]:
        base = pm25.rolling(w, min_periods=max(1, int(w * 0.5)))
        df[f"pm25_roll_mean_{w}"] = base.mean()
        df[f"pm25_roll_std_{w}"] = base.std()
        df[f"pm25_roll_max_{w}"] = base.max()
        df[f"pm25_roll_min_{w}"] = base.min()
    df["roll24_diff_24"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(24)
    df["roll24_diff_48"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(48)
    df["roll24_diff_1"] = df["pm25_roll_mean_24"] - df["pm25_roll_mean_24"].shift(1)
    df["roll24_accel"] = df["roll24_diff_24"] - df["roll24_diff_24"].shift(24)
    df["roll24_growth"] = df["pm25_roll_mean_24"] / (df["pm25_roll_mean_24"].shift(24) + 1e-5)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
        for lag in [1, 6, 12, 24, 48]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_roll24"] = df[col].rolling(24, min_periods=12).mean()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df.bfill().ffill().fillna(0.0)
    return df


def predict_24h(df_feat, artifact):
    latest = df_feat.iloc[-1:]
    X = latest[artifact["feature_cols"]].values
    Xs = artifact["scaler"].transform(X)
    ridge = float(artifact["ridge_model"].predict(Xs)[0])
    resid = float(artifact["residual_tree_model"].predict(X)[0])
    return max(5.0, ridge + resid), ridge, resid


def _secret_google_key():
    try:
        return str(st.secrets.get("GOOGLE_WEATHER_API_KEY", "") or "")
    except Exception:
        return ""


def fetch_google_weather(api_key):
    url = (
        "https://weather.googleapis.com/v1/currentConditions:lookup"
        f"?key={api_key}&location.latitude={DHAKA_LAT}&location.longitude={DHAKA_LON}"
        "&unitsSystem=METRIC"
    )
    data = http_json(url, timeout=15)
    wind = data.get("wind", {}).get("speed", {}) or {}
    precip = data.get("precipitation") or {}
    rain = ((precip.get("qpf") or {}).get("quantity"))
    rain_pct = (precip.get("probability") or {}).get("percent")
    return {
        "temperature": float(data["temperature"]["degrees"]),
        "humidity": float(data["relativeHumidity"]),
        "wind_speed": float(wind.get("value", 0.0)),
        "rainfall": float(rain or 0.0),
        "rain_chance_pct": None if rain_pct is None else float(rain_pct),
        "time": data.get("currentTime", ""),
        "source_name": "Google Weather — Dhaka",
        "verify_url": GOOGLE_WEATHER_PAGE,
    }


def fetch_weather_channel():
    url = (
        "https://api.weather.com/v3/wx/observations/current"
        f"?geocode={DHAKA_LAT},{DHAKA_LON}&units=m&language=en-US&format=json&apiKey={TWC_KEY}"
    )
    data = http_json(url, timeout=15)
    return {
        "temperature": float(data["temperature"]),
        "humidity": float(data["relativeHumidity"]),
        "wind_speed": float(data["windSpeed"]),
        "rainfall": float(data.get("precip1Hour") or 0.0),
        "rain_chance_pct": None,
        "time": data.get("validTimeLocal") or "",
        "source_name": "weather.com Today — Dhaka",
        "verify_url": WEATHERCOM_TODAY_PAGE,
    }


def fetch_weather_com_pm25():
    url = (
        "https://api.weather.com/v3/wx/globalAirQuality"
        f"?geocode={DHAKA_LAT},{DHAKA_LON}&language=en-US&scale=EPA&format=json&apiKey={TWC_KEY}"
    )
    data = http_json(url, timeout=15)
    g = data["globalairquality"]
    pm = float(g["pollutants"]["PM2.5"]["amount"])
    exp = g.get("expireTimeGmt")
    when = ""
    if exp:
        when = datetime.datetime.fromtimestamp(int(exp), tz=BST).strftime("%Y-%m-%d %H:%M")
    return {
        "value": pm,
        "aqi": g.get("airQualityIndex"),
        "category": g.get("airQualityCategory") or "",
        "time": when,
        "source_name": "weather.com Air Quality — Dhaka",
        "verify_url": WEATHERCOM_AQ_PAGE,
    }


def fetch_live_dhaka_data(google_api_key=""):
    url_hist_aq = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={DHAKA_LAT}&longitude={DHAKA_LON}"
        "&hourly=pm2_5&timezone=Asia%2FDhaka&past_days=14&forecast_days=1"
    )
    url_hist_wx = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={DHAKA_LAT}&longitude={DHAKA_LON}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,rain"
        "&timezone=Asia%2FDhaka&past_days=14&forecast_days=1"
    )
    data_aq = http_json(url_hist_aq)
    data_wx = http_json(url_hist_wx)

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(data_aq["hourly"]["time"]),
            "pm25": data_aq["hourly"]["pm2_5"],
            "temperature": data_wx["hourly"]["temperature_2m"],
            "humidity": data_wx["hourly"]["relative_humidity_2m"],
            "wind_speed": data_wx["hourly"]["wind_speed_10m"],
            "rainfall": data_wx["hourly"]["rain"],
        }
    )
    now = now_dhaka_naive()
    df = df[df["datetime"] <= now.floor("h")].dropna().sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No observed hourly history for Dhaka.")

    pm_meta = fetch_weather_com_pm25()
    df.loc[df.index[-1], "pm25"] = pm_meta["value"]

    used_weather = None
    weather_error = ""
    google_ok = False
    key = (google_api_key or "").strip()
    if key:
        try:
            used_weather = fetch_google_weather(key)
            google_ok = True
        except Exception as exc:
            weather_error = f"Google Weather API failed ({exc}). Using weather.com Today instead."

    if used_weather is None:
        used_weather = fetch_weather_channel()

    df.loc[df.index[-1], "temperature"] = used_weather["temperature"]
    df.loc[df.index[-1], "humidity"] = used_weather["humidity"]
    df.loc[df.index[-1], "wind_speed"] = used_weather["wind_speed"]
    df.loc[df.index[-1], "rainfall"] = used_weather["rainfall"]

    return df, {
        "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "pm": pm_meta,
        "weather": used_weather,
        "google_ok": google_ok,
        "weather_error": weather_error.strip(),
    }


def backtest_last_7_days(df, artifact):
    df_feat = engineer_live_features(df, artifact["feature_cols"])
    pm_vals = df["pm25"].to_numpy(dtype=float)
    n = len(df)
    end = n - 24
    if end <= 180:
        return pd.DataFrame()
    start = max(168, end - 168)
    X = df_feat.iloc[start:end][artifact["feature_cols"]].values
    Xs = artifact["scaler"].transform(X)
    pred = artifact["ridge_model"].predict(Xs) + artifact["residual_tree_model"].predict(X)
    pred = np.maximum(5.0, pred)
    actual = np.array([np.nanmean(pm_vals[i + 1 : i + 25]) for i in range(start, end)], dtype=float)
    persistence = np.array([np.nanmean(pm_vals[max(0, i - 23) : i + 1]) for i in range(start, end)], dtype=float)
    return pd.DataFrame(
        {
            "datetime": df["datetime"].iloc[start:end].to_numpy(),
            "predicted_24h_mean": pred,
            "actual_24h_mean": actual,
            "persistence": persistence,
        }
    ).dropna()


def metric_with_source(label, value, unit, source_name, when, verify_url):
    st.metric(label, f"{value} {unit}")
    st.caption(source_name)
    if when:
        st.caption(f"{when} (Dhaka)")
    st.markdown(f'<a href="{verify_url}" target="_blank">Open source 🔗</a>', unsafe_allow_html=True)


@st.cache_data(ttl=90, show_spinner="Fetching live Dhaka observations...")
def cached_live_fetch(google_key, nonce):
    return fetch_live_dhaka_data(google_key)


def render_live_panel(google_key, nonce):
    if artifact is None:
        st.error("Model failed to load.")
        return
    try:
        df_live, meta = cached_live_fetch(google_key, nonce)
        df_feat = engineer_live_features(df_live, artifact["feature_cols"])
        pred_24h, pred_ridge, pred_res = predict_24h(df_feat, artifact)
        bt = backtest_last_7_days(df_live, artifact)

        curr_pm = float(df_live["pm25"].iloc[-1])
        curr_temp = float(df_live["temperature"].iloc[-1])
        curr_hum = float(df_live["humidity"].iloc[-1])
        curr_wind = float(df_live["wind_speed"].iloc[-1])
        curr_rain = float(df_live["rainfall"].iloc[-1])

        st.caption(f"Live fetch · {meta['fetched_at']} BST · not a default value")
        if meta["weather_error"]:
            st.warning(meta["weather_error"])

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_with_source(
                "Current PM2.5",
                f"{curr_pm:.1f}",
                "µg/m³",
                meta["pm"]["source_name"],
                meta["pm"]["time"],
                meta["pm"]["verify_url"],
            )
        with c2:
            metric_with_source(
                "Temperature",
                f"{curr_temp:.1f}",
                "°C",
                meta["weather"]["source_name"],
                meta["weather"].get("time", ""),
                meta["weather"]["verify_url"],
            )
        with c3:
            metric_with_source(
                "Relative Humidity",
                f"{curr_hum:.0f}",
                "%",
                meta["weather"]["source_name"],
                meta["weather"].get("time", ""),
                meta["weather"]["verify_url"],
            )
        with c4:
            metric_with_source(
                "Wind Speed",
                f"{curr_wind:.1f}",
                "km/h",
                meta["weather"]["source_name"],
                meta["weather"].get("time", ""),
                meta["weather"]["verify_url"],
            )
        with c5:
            rain_src = meta["weather"]["source_name"] + " · mm last 1h"
            rain_pct = meta["weather"].get("rain_chance_pct")
            if rain_pct is not None:
                rain_src = f"{meta['weather']['source_name']} · {curr_rain:.1f} mm (card {rain_pct:.0f}% is chance, not mm)"
            metric_with_source(
                "Rainfall (1h)",
                f"{curr_rain:.1f}",
                "mm",
                rain_src,
                meta["weather"].get("time", ""),
                meta["weather"]["verify_url"],
            )

        if meta["pm"].get("aqi") is not None:
            st.caption(
                f"weather.com large number is AQI {meta['pm']['aqi']} ({meta['pm'].get('category','')}). "
                f"Match the PM2.5 µg/m³ line: {curr_pm:.1f}."
            )
        if not meta.get("google_ok"):
            st.info(f"No Google key — weather from weather.com Today. Key: {GOOGLE_DEMO_KEY_PAGE}")

        st.subheader("Predicted next-24h mean PM2.5")
        band_name, css_class, band_desc = get_aqi_band_info(pred_24h)
        st.markdown(
            f"""
<div class="{css_class}">
  <h2 style="margin:0;">{pred_24h:.1f} µg/m³</h2>
  <p style="margin:6px 0 0 0;">{band_name} · {band_desc}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"Ridge {pred_ridge:.1f} + residual {pred_res:+.1f} = {pred_24h:.1f} µg/m³")

        st.subheader("Last 7 days")
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=False)
        recent = df_live.iloc[-168:]
        axes[0].plot(recent["datetime"], recent["pm25"], color="#185FA5", lw=2.0, label="Observed hourly PM2.5")
        future_time = recent["datetime"].iloc[-1] + pd.Timedelta(hours=24)
        axes[0].plot(
            [recent["datetime"].iloc[-1], future_time],
            [recent["pm25"].iloc[-1], pred_24h],
            color="#D85A30",
            ls="--",
            lw=2.2,
            marker="o",
            label=f"24h-ahead mean ({pred_24h:.1f})",
        )
        axes[0].axhline(50, color="#2CA02C", ls=":", alpha=0.7, label="Good 50")
        axes[0].axhline(100, color="#FFC107", ls=":", alpha=0.7, label="Moderate 100")
        axes[0].set_title("Observed hourly PM2.5")
        axes[0].set_ylabel("PM2.5 (µg/m³)")
        axes[0].grid(True, ls="--", alpha=0.45)
        axes[0].legend(loc="upper left", fontsize=8)

        if not bt.empty:
            axes[1].plot(bt["datetime"], bt["actual_24h_mean"], color="#2CA02C", lw=2.0, label="Actual next-24h mean")
            axes[1].plot(bt["datetime"], bt["predicted_24h_mean"], color="#D85A30", lw=2.0, ls="--", label="Model")
            axes[1].plot(bt["datetime"], bt["persistence"], color="#7B8A9A", lw=1.4, ls=":", label="Naive last-24h mean")
            mae = float(np.mean(np.abs(bt["predicted_24h_mean"] - bt["actual_24h_mean"])))
            mae_p = float(np.mean(np.abs(bt["persistence"] - bt["actual_24h_mean"])))
            bias = float(np.mean(bt["predicted_24h_mean"] - bt["actual_24h_mean"]))
            axes[1].set_title(f"24h-ahead mean vs actual  (model MAE {mae:.1f}, naive {mae_p:.1f}, bias {bias:+.1f})")
            axes[1].set_ylabel("PM2.5 (µg/m³)")
            axes[1].grid(True, ls="--", alpha=0.45)
            axes[1].legend(loc="upper left", fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "Not enough completed 24h windows yet.", ha="center", va="center")
            axes[1].set_axis_off()
            mae = mae_p = bias = None

        axes[1].set_xlabel("Date/time (Asia/Dhaka)")
        fig.tight_layout()
        st.pyplot(fig)

        if not bt.empty:
            daily = bt.copy()
            daily["day"] = pd.to_datetime(daily["datetime"]).dt.date
            daily_tbl = (
                daily.groupby("day", as_index=False)
                .agg(predicted=("predicted_24h_mean", "mean"), actual=("actual_24h_mean", "mean"))
                .rename(columns={"day": "Date", "predicted": "Predicted 24h-mean", "actual": "Actual 24h-mean"})
            )
            daily_tbl["Abs. error"] = (daily_tbl["Predicted 24h-mean"] - daily_tbl["Actual 24h-mean"]).abs()
            st.dataframe(
                daily_tbl.style.format(
                    {"Predicted 24h-mean": "{:.1f}", "Actual 24h-mean": "{:.1f}", "Abs. error": "{:.1f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Target is the next-24h mean, not the next hourly reading. "
                f"Training mean ~88 µg/m³ (August ~36); current monsoon history ~17, so the model is biased high "
                f"(bias {bias:+.1f}). Thesis test R² is 0.86 on 2021–22, not this monsoon window."
            )
    except Exception as exc:
        st.error(f"Live fetch / forecast failed: {exc}")


with st.sidebar:
    st.markdown("**Dhaka only** · 23.8103° N, 90.4125° E")
    st.markdown("PM2.5 · Temperature · Humidity · Wind · Rainfall")
    st.markdown(
        f"""
**Sources**
- PM2.5: [weather.com Air Quality]({WEATHERCOM_AQ_PAGE})
- Weather: [Google Weather]({GOOGLE_WEATHER_PAGE})
"""
    )
    default_key = os.environ.get("GOOGLE_WEATHER_API_KEY", "") or _secret_google_key()
    google_key = st.text_input(
        "Google Weather API key",
        value=default_key,
        type="password",
        help="Needed for the four weather fields to match the Google Weather card.",
    )
    st.caption(f"Free demo key (no card): {GOOGLE_DEMO_KEY_PAGE}")
    auto_refresh = st.checkbox("Auto-refresh every 2 minutes", value=True)
    if st.button("Refresh now"):
        st.session_state["live_nonce"] = int(st.session_state.get("live_nonce", 0)) + 1
        cached_live_fetch.clear()
        st.rerun()


if "live_nonce" not in st.session_state:
    st.session_state["live_nonce"] = 0

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "System 1: Live Dhaka → 24h PM2.5",
        "System 2: Manual input",
        "System 3: CSV upload",
        "System 4: Citation",
    ]
)

with tab1:
    st.markdown(
        f"""
<div class="source-box">
<b>PM2.5:</b> <a href="{WEATHERCOM_AQ_PAGE}" target="_blank">weather.com Air Quality — Dhaka</a>
(use the PM2.5 µg/m³ line; the large number is AQI)<br>
<b>Temperature / humidity / wind / rain:</b>
<a href="{GOOGLE_WEATHER_PAGE}" target="_blank">Google Weather — Dhaka</a>
(requires the sidebar API key)
</div>
""",
        unsafe_allow_html=True,
    )

    nonce = int(st.session_state["live_nonce"])
    if auto_refresh and hasattr(st, "fragment"):

        @st.fragment(run_every=LIVE_REFRESH)
        def _auto_live():
            render_live_panel(google_key, nonce)

        _auto_live()
    else:
        render_live_panel(google_key, nonce)


with tab2:
    st.caption("Manual any-day inputs. The numbers below are form starters, not live readings.")
    col_a, col_b = st.columns(2)
    with col_a:
        man_date = st.date_input("Date", value=datetime.date.today())
        man_pm25 = st.number_input("PM2.5 (µg/m³)", value=25.0, step=1.0)
        man_pm25_lag24 = st.number_input("PM2.5 24h earlier (µg/m³)", value=22.0, step=1.0)
        man_temp = st.number_input("Temperature (°C)", value=30.0, step=0.5)
    with col_b:
        man_hum = st.number_input("Relative humidity (%)", value=80.0, step=1.0)
        man_wind = st.number_input("Wind speed (km/h)", value=15.0, step=0.5)
        man_rain = st.number_input("Rainfall (mm)", value=0.0, step=0.1)

    if st.button("Predict 24h PM2.5", type="primary"):
        if artifact is None:
            st.error("Model failed to load.")
        else:
            df_sim = pd.DataFrame(
                {
                    "datetime": pd.date_range(end=pd.Timestamp(man_date) + pd.Timedelta(hours=23), periods=240, freq="1h"),
                    "pm25": np.linspace(man_pm25_lag24, man_pm25, 240),
                    "temperature": [man_temp] * 240,
                    "humidity": [man_hum] * 240,
                    "wind_speed": [man_wind] * 240,
                    "rainfall": [0.0] * 239 + [man_rain],
                }
            )
            df_sim_feat = engineer_live_features(df_sim, artifact["feature_cols"])
            sim_pred, sim_ridge, sim_res = predict_24h(df_sim_feat, artifact)
            band_name, css_class, band_desc = get_aqi_band_info(sim_pred)
            st.markdown(
                f"""
<div class="{css_class}">
  <h2 style="margin:0;">{sim_pred:.1f} µg/m³</h2>
  <p style="margin:6px 0 0 0;">{band_name} · {band_desc}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.caption(f"Ridge {sim_ridge:.1f} + residual {sim_res:+.1f} = {sim_pred:.1f} µg/m³")


with tab3:
    st.caption("Required columns: datetime, pm25, temperature, humidity, wind_speed, rainfall")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    if uploaded_file is not None and artifact is not None:
        df_custom = pd.read_csv(uploaded_file)
        df_custom["datetime"] = pd.to_datetime(df_custom["datetime"])
        st.write(df_custom.head())
        if st.button("Run bulk 24h-ahead forecast"):
            df_cust_feat = engineer_live_features(df_custom, artifact["feature_cols"])
            X_cust = df_cust_feat[artifact["feature_cols"]].values
            X_cust_s = artifact["scaler"].transform(X_cust)
            p_cust = np.maximum(
                5.0,
                artifact["ridge_model"].predict(X_cust_s) + artifact["residual_tree_model"].predict(X_cust),
            )
            df_custom["predicted_pm25_24h_avg"] = p_cust
            st.dataframe(df_custom[["datetime", "pm25", "predicted_pm25_24h_avg"]].tail(20))
            st.download_button(
                "Download predictions CSV",
                data=df_custom.to_csv(index=False).encode("utf-8"),
                file_name="custom_dhaka_pm25_forecasts.csv",
                mime="text/csv",
            )


with tab4:
    st.code(
        """@article{DhakaPM25Forecast2026,
  title={Multivariate 24-Hour Ahead Daily-Average PM2.5 Forecasting in Dhaka Using Hybrid Ridge-Residual Gradient Boosting},
  author={Talukder, Meherab Hossain and Collaborators},
  journal={Journal of Environmental Management / Atmospheric Environment},
  year={2026},
  note={https://github.com/meherabmehu/PM2.5-Air-Quality-Forecasting-for-Dhaka-}
}""",
        language="bibtex",
    )
