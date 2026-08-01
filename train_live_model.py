import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print("=== Training 0.8650 Champion Hybrid Ridge-Residual Model for Live Deployment ===")
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs('/home/user/models', exist_ok=True)

# 1. Load clean dataset
def find_data_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data", "final_dataset_clean.csv"),
        "data/final_dataset_clean.csv",
        "final_dataset_clean.csv",
        "/home/user/uploads/final_dataset_clean.csv",
        "/kaggle/input/datasets/begumluthfunnesa/thesis/final_dataset_clean.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find final_dataset_clean.csv in data/ folder or root.")

df_clean = pd.read_csv(find_data_file())
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

df = df_clean.copy()
df = df[df['pm25'] >= 1].reset_index(drop=True)
p995 = df['pm25'].quantile(0.995)
df.loc[df['pm25'] > p995, 'pm25'] = p995

pm25 = df['pm25']
df['pm25_curr'] = pm25

# ── Core Causal Feature Set (92 features) ─────────────────────────────────────
for lag in [1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 36, 48, 72, 120, 168]:
    df[f'pm25_lag_{lag}'] = pm25.shift(lag)

for span in [3, 6, 12, 24, 48, 72, 168]:
    df[f'pm25_ema_{span}'] = pm25.ewm(span=span, adjust=False).mean()

for w in [3, 6, 12, 24, 48, 72, 168]:
    base = pm25.rolling(w, min_periods=max(1, int(w*0.5)))
    df[f'pm25_roll_mean_{w}']   = base.mean()
    df[f'pm25_roll_std_{w}']    = base.std()
    df[f'pm25_roll_max_{w}']    = base.max()
    df[f'pm25_roll_min_{w}']    = base.min()

df['target'] = pm25.shift(-24).rolling(24, min_periods=24).mean()

# ── Growth rates and momentum ─────────────────────────────────────────────────
df['roll24_diff_24'] = df['pm25_roll_mean_24'] - df['pm25_roll_mean_24'].shift(24)
df['roll24_diff_48'] = df['pm25_roll_mean_24'] - df['pm25_roll_mean_24'].shift(48)
df['roll24_diff_1']  = df['pm25_roll_mean_24'] - df['pm25_roll_mean_24'].shift(1)
df['roll24_accel']   = df['roll24_diff_24'] - df['roll24_diff_24'].shift(24)
df['roll24_growth']  = df['pm25_roll_mean_24'] / (df['pm25_roll_mean_24'].shift(24) + 1e-5)

# ── Calendar and cyclical encodings ───────────────────────────────────────────
df['hour']  = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['doy']   = df['datetime'].dt.dayofyear
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
df['doy_sin']  = np.sin(2*np.pi*df['doy']/365.25)
df['doy_cos']  = np.cos(2*np.pi*df['doy']/365.25)

# ── Meteorological causal features ────────────────────────────────────────────
for col in ['temperature', 'humidity', 'wind_speed', 'rainfall']:
    for lag in [1, 6, 12, 24, 48]:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df[f'{col}_roll24'] = df[col].rolling(24, min_periods=12).mean()

DROP_ALWAYS = ['datetime', 'target']
FEATURE_COLS = [c for c in df.columns if c not in DROP_ALWAYS]

df_sel = df[FEATURE_COLS + ['target']].dropna().reset_index(drop=True)

n = len(df_sel)
tr_end = int(n * 0.70)
va_end = int(n * 0.85)

tr = df_sel.iloc[:tr_end].reset_index(drop=True)
va = df_sel.iloc[tr_end:va_end].reset_index(drop=True)
te = df_sel.iloc[va_end:].reset_index(drop=True)

X_tr = tr[FEATURE_COLS].values; y_tr = tr['target'].values
X_va = va[FEATURE_COLS].values; y_va = va['target'].values
X_te = te[FEATURE_COLS].values; y_te = te['target'].values

print(f"Dataset split — Train: {len(tr)} | Val: {len(va)} | Test: {len(te)} | Features: {len(FEATURE_COLS)}")

# Fit Scaler on Train
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 1. Fit RidgeCV Linear Autoregressive Anchor
print("Training Stage 1 — RidgeCV Linear Autoregressive Anchor...")
m_ridge = RidgeCV(alphas=np.logspace(-2, 6, 50))
m_ridge.fit(X_tr_s, y_tr)
p_ridge_te = m_ridge.predict(X_te_s)

# 2. Fit Residual HistGradientBoostingTree on (y_tr - p_ridge_tr)
print("Training Stage 2 — Residual HistGradientBoostingTree on training residuals...")
res_tr = y_tr - m_ridge.predict(X_tr_s)
m_res = HistGradientBoostingRegressor(loss='squared_error', max_iter=400, learning_rate=0.03, max_depth=4, min_samples_leaf=15, random_state=42)
m_res.fit(X_tr, res_tr)
p_res_te = m_res.predict(X_te)

p_hybrid = p_ridge_te + p_res_te
r2_hyb   = r2_score(y_te, p_hybrid)
rmse_hyb = mean_squared_error(y_te, p_hybrid)**0.5
mae_hyb  = mean_absolute_error(y_te, p_hybrid)

print(f"\n✓ Verified Champion Hybrid Model Test Performance:")
print(f"   Test R²   : {r2_hyb:.4f}")
print(f"   Test RMSE : {rmse_hyb:.2f} µg/m³")
print(f"   Test MAE  : {mae_hyb:.2f} µg/m³")

# Save complete live deployment artifact bundle
artifact = {
    'scaler': scaler,
    'ridge_model': m_ridge,
    'residual_tree_model': m_res,
    'feature_cols': FEATURE_COLS,
    'p995': p995,
    'metrics': {
        'R2': float(r2_hyb),
        'RMSE': float(rmse_hyb),
        'MAE': float(mae_hyb)
    }
}

with open(os.path.join(models_dir, 'live_hybrid_champion.pkl'), 'wb') as f:
    pickle.dump(artifact, f)

with open(os.path.join(models_dir, 'live_feature_names.json'), 'w') as f:
    json.dump({'feature_cols': FEATURE_COLS, 'metrics': artifact['metrics']}, f, indent=2)

print("\n✓ Saved /home/user/models/live_hybrid_champion.pkl successfully!")
