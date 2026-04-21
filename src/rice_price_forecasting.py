"""
Rice Price Forecasting Pipeline
=================================
Hybrid QGA–QPSO: XGBoost (trend) + LSTM (residuals)

Pipeline Steps:
  1. Descriptive Statistics
  2. ACF & Periodogram Analysis
  3. Column Normalization
  4. Feature Engineering
  5. Train/Val/Test Split
  6. Baseline ML & DL Evaluation
  7. Quantum Feature Selection (QGA / QACO)
  8. QPSO Hyperparameter Optimization
  9. Hybrid Model Training & Evaluation
 10. Multi-Horizon Forecast Performance
"""

# ============================================================
# IMPORTS
# ============================================================
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import periodogram
from scipy.stats import jarque_bera, kurtosis, shapiro, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# STEP 1 — DESCRIPTIVE STATISTICS
# ============================================================

def compute_descriptive_stats(filepath: str) -> pd.DataFrame:
    """Load data and compute per-state descriptive statistics."""
    df = pd.read_excel(filepath)
    df["Modal Price"] = pd.to_numeric(df["Modal Price"], errors="coerce")
    df = df.dropna(subset=["STATE_KEY", "Modal Price"])

    rows = []
    for state, g in df.groupby("STATE_KEY"):
        prices = g["Modal Price"].values
        jb_stat, jb_p = jarque_bera(prices)
        sh_stat, sh_p = shapiro(prices[:5000])   # Shapiro limit safeguard
        rows.append({
            "State":       state,
            "Min":         prices.min(),
            "Mean":        prices.mean(),
            "Max":         prices.max(),
            "Std Dev":     prices.std(),
            "CV (%)":      (prices.std() / prices.mean()) * 100,
            "Skewness":    skew(prices),
            "Kurtosis":    kurtosis(prices, fisher=False),
            "JB test":     f"{jb_stat:.2f} ({jb_p:.4f})",
            "Shapiro-Wilk": f"{sh_stat:.2f} ({sh_p:.4f})",
        })

    desc_df = pd.DataFrame(rows).round(2)
    print(desc_df)
    return desc_df


# ============================================================
# STEP 2 — ACF & PERIODOGRAM ANALYSIS
# ============================================================

# Settings
USE_NATIONAL_SERIES = True
SELECT_STATES       = ["Karnataka", "Tamil Nadu", "Punjab", "Andhra Pradesh", "West Bengal"]
DATE_COL            = "DATE_STD"
STATE_COL           = "STATE_KEY"
TARGET_COL          = "Modal Price"
MAX_LAGS            = 60
FIG_DPI             = 200
FREQ                = "W"      # "W" = weekly | "D" = daily
OUT_ACF             = "Fig_ACF.png"
OUT_PERIODOGRAM     = "Fig_Periodogram_DoubleLog.png"


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)

    # Normalize column names (strip BOM, non-breaking spaces, whitespace)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

    df[DATE_COL]   = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL, STATE_COL]).copy()
    df = df.drop_duplicates(subset=[STATE_COL, DATE_COL], keep="last")
    df = df.sort_values([STATE_COL, DATE_COL]).reset_index(drop=True)

    if DATE_COL not in df.columns:
        raise KeyError(f"{DATE_COL} not found. Available: {df.columns.tolist()}")

    print(f"Rows after cleaning: {len(df)}")
    print(f"Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}")
    print(f"States: {df[STATE_COL].nunique()}")
    return df


def make_national_series(df: pd.DataFrame) -> pd.Series:
    return df.groupby(DATE_COL)[TARGET_COL].mean().sort_index()


def make_state_series(df: pd.DataFrame, state: str) -> pd.Series:
    return (
        df[df[STATE_COL] == state]
        .set_index(DATE_COL)[TARGET_COL]
        .sort_index()
    )


def regularize(s: pd.Series, freq: str) -> pd.Series:
    """Regularize series to a uniform frequency; interpolate gaps."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.asfreq(freq)
    s = s.interpolate(method="time", limit_direction="both")
    return s


def make_stationary(s: pd.Series):
    """Return log-returns (or first-diff if non-positive prices exist)."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if (s <= 0).any():
        return s.diff().dropna(), "ΔPrice (1st diff)"
    return np.log(s).diff().dropna(), "Δlog(Price) (log-return)"


def build_series(df: pd.DataFrame):
    series_list, series_names = [], []
    if USE_NATIONAL_SERIES:
        series_list  = [make_national_series(df)]
        series_names = ["National_Average"]
    else:
        available = set(df[STATE_COL].unique())
        for st in SELECT_STATES:
            if st not in available:
                print(f"⚠️  State not found: {st}")
                continue
            series_list.append(make_state_series(df, st))
            series_names.append(st)

    if not series_list:
        raise ValueError("No series available. Check state names or dataset columns.")

    series_list = [regularize(s, FREQ) for s in series_list]
    return series_list, series_names


def plot_acf_figure(stationary_list, series_names, stationary_label):
    plt.figure(figsize=(7, 4))
    unit = "weeks" if FREQ == "W" else "days"
    if len(stationary_list) == 1:
        plot_acf(
            stationary_list[0].values,
            lags=MAX_LAGS,
            ax=plt.gca(),
            title=f"ACF: {series_names[0]} ({stationary_label})",
        )
        plt.xlabel(f"Lag ({unit})")
    else:
        for s_stat, name in zip(stationary_list, series_names):
            acf_vals = acf(s_stat.values, nlags=MAX_LAGS, fft=True)
            plt.plot(range(len(acf_vals)), acf_vals, label=name)
        plt.axhline(0, linewidth=1)
        plt.title(f"ACF: {stationary_label} (Selected States)")
        plt.xlabel(f"Lag ({unit})")
        plt.ylabel("ACF")
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ACF, dpi=FIG_DPI)
    plt.show()
    plt.close()
    print(f"✅ Saved: {OUT_ACF}")


def plot_periodogram(stationary_list, series_names, stationary_label):
    plt.figure(figsize=(7, 4))
    fs = 1.0
    for s_stat, name in zip(stationary_list, series_names):
        x = s_stat.values.astype(float) - s_stat.values.mean()
        freqs, pxx = periodogram(x, fs=fs, scaling="density", detrend="linear")
        freqs, pxx = freqs[1:], pxx[1:]
        pxx = np.where(pxx <= 0, np.nan, pxx)
        plt.loglog(freqs, pxx, label=name)

    unit = "week" if FREQ == "W" else "day"
    plt.title(f"Periodogram (Double-Log): {stationary_label}")
    plt.xlabel(f"Frequency (cycles/{unit}) [log]")
    plt.ylabel("Power Spectral Density [log]")
    if len(stationary_list) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PERIODOGRAM, dpi=FIG_DPI)
    plt.show()
    plt.close()
    print(f"✅ Saved: {OUT_PERIODOGRAM}")

    # Dominant period
    long_thresh = 52 if FREQ == "W" else 365
    for s_stat, name in zip(stationary_list, series_names):
        x = s_stat.values.astype(float) - s_stat.values.mean()
        freqs, pxx = periodogram(x, fs=fs, scaling="density", detrend="linear")
        freqs, pxx = freqs[1:], pxx[1:]
        mask = freqs >= (1 / long_thresh)
        f2, p2 = freqs[mask], pxx[mask]
        if len(p2) > 0:
            f_dom = f2[np.nanargmax(p2)]
            steps = 1.0 / f_dom
            print(f"Dominant period for {name}: ~{steps:.1f} {'weeks' if FREQ=='W' else 'days'}")


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

TARGET     = "Modal Price"
BASE_TEMP  = 10.0   # rice base temperature (°C)
PRICE_LAGS = [1, 3, 7, 14, 30]
ROLL_WIN   = [7, 14, 30]
CLIMATE_LAGS = [7, 14]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE_STD"] = pd.to_datetime(df["DATE_STD"], dayfirst=True, errors="coerce")
    df = df.sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)

    # Log target
    df["y"] = np.log1p(df[TARGET])

    # Price lags
    for lag in PRICE_LAGS:
        df[f"price_lag_{lag}"] = df.groupby("STATE_KEY")[TARGET].shift(lag)

    # Rolling statistics (shift first to avoid leakage)
    for w in ROLL_WIN:
        shifted = df.groupby("STATE_KEY")[TARGET].shift(1)
        df[f"price_ma_{w}"]  = shifted.rolling(w).mean()
        df[f"price_std_{w}"] = shifted.rolling(w).std()

    # Calendar features
    df["month"]      = df["DATE_STD"].dt.month
    df["weekofyear"] = df["DATE_STD"].dt.isocalendar().week.astype(int)
    df["quarter"]    = df["DATE_STD"].dt.quarter
    df["dayofyear"]  = df["DATE_STD"].dt.dayofyear

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Growing-degree days
    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        tmean      = (df["T2M_MAX"] + df["T2M_MIN"]) / 2
        df["GDD"]  = np.maximum(tmean - BASE_TEMP, 0.0)

    # Climate lags
    for lag in CLIMATE_LAGS:
        for col in ["PRECTOTCORR", "T2M_MAX", "T2M_MIN"]:
            if col in df.columns:
                df[f"{col}_lag_{lag}"] = df.groupby("STATE_KEY")[col].shift(lag)

    # Additional rolling std for price
    for lag in [7, 14]:
        if "T2M_MAX" in df.columns:
            df[f"T2M_MAX_lag_{lag}"] = df["T2M_MAX"].shift(lag)
        if "T2M_MIN" in df.columns:
            df[f"T2M_MIN_lag_{lag}"] = df["T2M_MIN"].shift(lag)

    df["price_std_14"] = df[TARGET].rolling(14).std()
    df["price_std_30"] = df[TARGET].rolling(30).std()

    # Drop NaNs and save
    FEATURE_COLS = [
        c for c in df.columns if c not in ["STATE_KEY", "DATE_STD", TARGET, "y"]
    ]
    df = df.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)

    # Save
    df[["STATE_KEY", "DATE_STD"] + FEATURE_COLS + ["y"]].to_parquet(
        "X_full.parquet", index=False
    )
    with open("feature_list_full.txt", "w") as f:
        f.writelines(c + "\n" for c in FEATURE_COLS)

    print(f"✅ Rows after FE: {len(df)}")
    print(f"✅ Saved: X_full.parquet, feature_list_full.txt")
    return df


# ============================================================
# STEP 4 — TRAIN / VAL / TEST SPLIT
# ============================================================

TEST_FRACTION     = 0.20
MIN_ROWS_STATE    = 120
N_FOLDS           = 10
VAL_BLOCK_FRAC    = 0.10
MIN_VAL_BLOCK     = 30


def split_statewise(df: pd.DataFrame, test_frac=0.2, min_rows=120):
    df = df.sort_values(["STATE_KEY", "DATE_STD"]).copy()
    tr_parts, te_parts, rows = [], [], []
    for st, g in df.groupby("STATE_KEY", sort=False):
        g = g.sort_values("DATE_STD")
        n = len(g)
        if n < min_rows:
            rows.append((st, n, None, None, None, None, "SKIPPED_SMALL"))
            continue
        cut = int((1 - test_frac) * n)
        trv, te = g.index[:cut], g.index[cut:]
        tr_parts.append(trv)
        te_parts.append(te)
        rows.append((
            st, n,
            g.loc[trv, "DATE_STD"].min(), g.loc[trv, "DATE_STD"].max(),
            g.loc[te,  "DATE_STD"].min(), g.loc[te,  "DATE_STD"].max(),
            "OK",
        ))

    trainval_idx = np.concatenate(tr_parts) if tr_parts else np.array([], dtype=int)
    test_idx     = np.concatenate(te_parts) if te_parts else np.array([], dtype=int)
    summary = pd.DataFrame(
        rows,
        columns=["STATE_KEY", "ROWS", "TRAINVAL_START", "TRAINVAL_END",
                 "TEST_START", "TEST_END", "STATUS"],
    )
    return trainval_idx, test_idx, summary


def build_expanding_folds(df, trainval_idx, n_folds=10, val_frac=0.1, min_val=30):
    df_tv = df.loc[trainval_idx].sort_values(["STATE_KEY", "DATE_STD"]).copy()
    state_to_idx = {st: g.index.to_numpy() for st, g in df_tv.groupby("STATE_KEY", sort=False)}

    folds, fold_summaries = [], []
    for k in range(n_folds):
        tr_parts, va_parts, info = [], [], []
        for st, idx in state_to_idx.items():
            n = len(idx)
            if n < min_val * 2:
                continue
            val_len  = max(min_val, int(val_frac * n))
            max_cut  = n - val_len
            train_end = int((k + 1) * (max_cut / (n_folds + 1)))
            train_end = max(min_val, min(train_end, max_cut))
            tr = idx[:train_end]
            va = idx[train_end:train_end + val_len]
            if len(tr) < min_val or len(va) < min_val:
                continue
            tr_parts.append(tr)
            va_parts.append(va)
            info.append((st, len(tr), len(va),
                         df.loc[tr, "DATE_STD"].min(), df.loc[tr, "DATE_STD"].max(),
                         df.loc[va, "DATE_STD"].min(), df.loc[va, "DATE_STD"].max()))

        if not tr_parts:
            continue
        fold_train = np.concatenate(tr_parts)
        fold_val   = np.concatenate(va_parts)
        folds.append({"train_idx": fold_train, "val_idx": fold_val})
        fold_summaries.append(pd.DataFrame(
            info,
            columns=["STATE_KEY", "TRAIN_N", "VAL_N",
                     "TRAIN_START", "TRAIN_END", "VAL_START", "VAL_END"],
        ))
        print(f"Fold {k+1}/{n_folds}: train={len(fold_train)} val={len(fold_val)} states={len(info)}")

    return folds, fold_summaries


def internal_split_statewise(df_tv, val_frac=0.2, min_rows=120):
    tr_parts, va_parts = [], []
    for st, g in df_tv.groupby("STATE_KEY", sort=False):
        g = g.sort_values("DATE_STD")
        if len(g) < min_rows:
            continue
        cut = int((1 - val_frac) * len(g))
        tr_parts.append(g.iloc[:cut])
        va_parts.append(g.iloc[cut:])
    return (
        pd.concat(tr_parts).reset_index(drop=True),
        pd.concat(va_parts).reset_index(drop=True),
    )


# ============================================================
# STEP 5 — METRICS & HELPERS
# ============================================================

EPS = 1e-9


def mape_safe(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(np.abs(y_true) + np.abs(y_pred) < EPS, EPS,
                     np.abs(y_true) + np.abs(y_pred))
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = max(np.sum(np.abs(y_true)), EPS)
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / denom


def eval_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":     float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%":  float(mape_safe(y_true, y_pred)),
        "SMAPE_%": float(smape(y_true, y_pred)),
        "WAPE_%":  float(wape(y_true, y_pred)),
        "MedAE":   float(np.median(np.abs(y_true - y_pred))),
        "R2":      float(r2_score(y_true, y_pred)),
    }


def get_Xy(df_part, feature_cols=None):
    drop = ["STATE_KEY", "DATE_STD", "y"]
    if feature_cols is None:
        feature_cols = [c for c in df_part.columns if c not in drop]
    X = df_part[feature_cols].to_numpy(dtype=np.float32)
    y = df_part["y"].to_numpy(dtype=np.float32)
    return X, y


# ============================================================
# STEP 6 — BASELINE MODEL DEFINITIONS
# ============================================================

def make_ml_model(name: str):
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=42)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=600, random_state=42,
                                     n_jobs=-1, min_samples_leaf=3)
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=5000, learning_rate=0.02,
                             num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=2.0,
                             random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=3000, learning_rate=0.03,
                            max_depth=8, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=42, n_jobs=-1, tree_method="hist")
    raise ValueError(f"Unknown model: {name}")


def make_univariate_sequences(y: np.ndarray, seq_len: int = 30):
    Xs, ys = [], []
    for i in range(seq_len, len(y)):
        Xs.append(y[i - seq_len:i])
        ys.append(y[i])
    Xs = np.array(Xs, dtype=np.float32).reshape(-1, seq_len, 1)
    ys = np.array(ys, dtype=np.float32)
    return Xs, ys


def make_seq(X: np.ndarray, y: np.ndarray, seq_len: int = 14):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_seq_X(X: np.ndarray, seq_len: int):
    return np.asarray(
        [X[i - seq_len:i] for i in range(seq_len, len(X))],
        dtype=np.float32,
    )


def build_lstm(seq_len: int, n_features: int = 1) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x   = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.LSTM(32)(x)
    x   = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_gru(seq_len: int, n_features: int = 1) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x   = tf.keras.layers.GRU(64)(inp)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# ============================================================
# STEP 7 — QUANTUM FEATURE SELECTION (QGA & QACO)
# ============================================================

# Lightweight XGB for fast fitness evaluation
XGB_EVAL = dict(
    n_estimators=800, learning_rate=0.05, max_depth=6,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
    random_state=42, n_jobs=-1, tree_method="hist",
)
LAMBDA_FEAT = 0.02   # feature-count penalty weight

# These are set at runtime after loading data
Xtr_full = Xva_full = ytr = yva = None


def fs_objective(mask: np.ndarray) -> float:
    """Fitness = RMSE + λ * feature_ratio."""
    k = int(mask.sum())
    if k == 0:
        return 1e9
    cols = np.where(mask == 1)[0]
    model = XGBRegressor(**XGB_EVAL)
    model.fit(Xtr_full[:, cols], ytr)
    pred  = model.predict(Xva_full[:, cols])
    rmse  = np.sqrt(mean_squared_error(yva, pred))
    return float(rmse + LAMBDA_FEAT * (k / Xtr_full.shape[1]))


# --- QGA ---

def qga_init(pop: int, d: int, p0: float = 0.5) -> np.ndarray:
    return np.full((pop, d), p0, dtype=np.float64)


def qga_measure(P: np.ndarray, rng) -> np.ndarray:
    return (rng.random(P.shape) < P).astype(np.int8)


def qga_update(P: np.ndarray, elite_masks: np.ndarray, lr: float = 0.12) -> np.ndarray:
    target = elite_masks.mean(axis=0)
    P = (1 - lr) * P + lr * target
    return np.clip(P, 0.02, 0.98)


def run_qga(pop: int = 28, gens: int = 18, elite_k: int = 6, seed: int = 42):
    rng = np.random.default_rng(seed)
    d   = Xtr_full.shape[1]
    P   = qga_init(pop, d)
    best_mask, best_score = None, 1e18

    for g in range(gens):
        masks  = qga_measure(P, rng)
        scores = np.array([fs_objective(m) for m in masks], dtype=float)
        top_k  = np.argsort(scores)[:elite_k]

        if scores[top_k[0]] < best_score:
            best_score = float(scores[top_k[0]])
            best_mask  = masks[top_k[0]].copy()

        P = qga_update(P, masks[top_k])
        print(f"[QGA] Gen {g+1}/{gens} | Best={best_score:.5f} | Feats={best_mask.sum()}/{d}")

    return best_mask, best_score


# --- QACO ---

def run_qaco(n_ants: int = 20, iters: int = 12,
             select_ratio: float = 0.35, evap: float = 0.25, seed: int = 42):
    rng  = np.random.default_rng(seed)
    d    = Xtr_full.shape[1]
    pher = np.ones(d, dtype=np.float64)
    best_mask, best_score = None, 1e18

    for it in range(iters):
        masks, scores = [], []
        for _ in range(n_ants):
            probs  = pher / pher.sum()
            k      = max(1, int(select_ratio * d))
            chosen = rng.choice(d, size=k, replace=False, p=probs)
            mask   = np.zeros(d, dtype=np.int8)
            mask[chosen] = 1
            score  = fs_objective(mask)
            masks.append(mask)
            scores.append(score)
            if score < best_score:
                best_score = float(score)
                best_mask  = mask.copy()

        pher = (1 - evap) * pher
        top  = np.argsort(scores)[:max(3, n_ants // 5)]
        for idx in top:
            pher += (1.0 / (scores[idx] + EPS)) * masks[idx]

        print(f"[QACO] Iter {it+1}/{iters} | Best={best_score:.5f} | Feats={best_mask.sum()}/{d}")

    return best_mask, best_score


# ============================================================
# STEP 8 — QPSO HYPERPARAMETER OPTIMISATION
# ============================================================

def vec_to_xgb_params(v: np.ndarray) -> dict:
    """Map [0,1]^7 → XGBoost hyperparameter dict."""
    return dict(
        learning_rate    = float(0.01 + v[0] * (0.15 - 0.01)),
        max_depth        = int(3   + v[1] * (12 - 3)),
        subsample        = float(0.6  + v[2] * (0.95 - 0.6)),
        colsample_bytree = float(0.6  + v[3] * (0.95 - 0.6)),
        reg_lambda       = float(0.0  + v[4] * (10.0 - 0.0)),
        min_child_weight = float(1.0  + v[5] * (10.0 - 1.0)),
        n_estimators     = int(500 + v[6] * (5000 - 500)),
        random_state=42, n_jobs=-1, tree_method="hist",
    )


def qpso_fitness(v, Xtr, ytr, Xva, yva):
    params = vec_to_xgb_params(v)
    model  = XGBRegressor(**params)
    model.fit(Xtr, ytr)
    pred   = model.predict(Xva)
    return float(np.sqrt(mean_squared_error(yva, pred)))


def run_qpso(Xtr, ytr, Xva, yva,
             n_particles: int = 20, iters: int = 30,
             beta: float = 0.75, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    dim = 7

    # Initialise particles uniformly in [0,1]^dim
    theta     = rng.random((n_particles, dim))
    pbest     = theta.copy()
    pbest_fit = np.array([qpso_fitness(p, Xtr, ytr, Xva, yva) for p in pbest])
    gbest     = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = float(pbest_fit.min())

    for t in range(iters):
        mbest = pbest.mean(axis=0)
        for i in range(n_particles):
            r      = rng.random(dim)
            p_att  = r * pbest[i] + (1 - r) * gbest          # local attractor
            u      = rng.random(dim)
            sign   = np.where(rng.random(dim) < 0.5, 1, -1)
            theta[i] = p_att + sign * beta * np.abs(mbest - theta[i]) * np.log(1.0 / (u + EPS))
            theta[i] = np.clip(theta[i], 0.0, 1.0)           # boundary constraints

            fit = qpso_fitness(theta[i], Xtr, ytr, Xva, yva)
            if fit < pbest_fit[i]:
                pbest[i], pbest_fit[i] = theta[i].copy(), fit

        gbest     = pbest[np.argmin(pbest_fit)].copy()
        gbest_fit = float(pbest_fit.min())
        print(f"[QPSO] Iter {t+1}/{iters} | BestRMSE={gbest_fit:.5f}")

    best_params = vec_to_xgb_params(gbest)
    print("✅ QPSO best params:", best_params)
    return best_params


# ============================================================
# STEP 9 — HYBRID MODEL (XGBoost trend + LSTM residuals)
# ============================================================

SEQ_LEN   = 14
HORIZONS  = [7, 14, 30]


def train_hybrid(df_train, df_val, df_test, feature_cols):
    """Train hybrid XGBoost + LSTM and return test metrics."""
    Xtr, ytr = get_Xy(df_train, feature_cols)
    Xva, yva = get_Xy(df_val,   feature_cols)
    Xte, yte = get_Xy(df_test,  feature_cols)

    # — Trend model —
    xgb_trend = make_ml_model("XGBoost")
    xgb_trend.fit(Xtr, ytr)

    train_hat = xgb_trend.predict(Xtr)
    val_hat   = xgb_trend.predict(Xva)
    test_hat  = xgb_trend.predict(Xte)

    # — Residuals —
    r_tr = (ytr - train_hat).astype(np.float32)
    r_va = (yva - val_hat).astype(np.float32)

    Xr_tr, yr_tr = make_seq(Xtr, r_tr, SEQ_LEN)
    Xr_va, yr_va = make_seq(Xva, r_va, SEQ_LEN)

    tf.keras.backend.clear_session()
    res_model = build_lstm(SEQ_LEN, n_features=Xtr.shape[1])
    res_model.fit(
        Xr_tr, yr_tr,
        validation_data=(Xr_va, yr_va),
        epochs=30, batch_size=256, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True)],
    )

    # — Hybrid test predictions —
    X_seq_test = make_seq_X(Xte, SEQ_LEN)
    resid_pred = res_model.predict(X_seq_test, verbose=0).reshape(-1)

    trend_aligned = test_hat[SEQ_LEN:]
    y_true        = yte[SEQ_LEN:]
    hybrid_pred   = trend_aligned + resid_pred

    met = eval_metrics(y_true, hybrid_pred)
    print("✅ Hybrid test metrics:", met)
    return xgb_trend, res_model, met


# ============================================================
# STEP 10 — MULTI-HORIZON FORECAST EVALUATION
# ============================================================

def evaluate_horizons(df_test, feature_cols, xgb_trend, res_model):
    """True multi-step horizon evaluation."""
    df_test_s = df_test.sort_values("DATE_STD").reset_index(drop=True)
    Xte, yte  = get_Xy(df_test_s, feature_cols)

    trend_test = xgb_trend.predict(Xte)
    X_seq_test = make_seq_X(Xte, SEQ_LEN)
    resid_all  = res_model.predict(X_seq_test, verbose=0).reshape(-1)

    y_true_all   = yte[SEQ_LEN:]
    trend_aligned = trend_test[SEQ_LEN:]
    pred_all     = trend_aligned + resid_all

    rows = []
    for H in HORIZONS:
        if H >= len(y_true_all):
            print(f"⚠️  Horizon {H} > test length {len(y_true_all)}. Skipping.")
            continue
        y_true_H = y_true_all[H:]
        y_pred_H = pred_all[:-H]
        rmse = np.sqrt(mean_squared_error(y_true_H, y_pred_H))
        mape = mape_safe(y_true_H, y_pred_H)
        rows.append({"Horizon (days ahead)": H, "RMSE": rmse,
                     "MAPE (%)": mape, "N_eval": len(y_true_H)})

    df_hor = pd.DataFrame(rows)
    print(df_hor)

    plt.figure(figsize=(6, 4))
    plt.plot(df_hor["Horizon (days ahead)"], df_hor["RMSE"],
             marker="o", linewidth=2, label="RMSE")
    plt.plot(df_hor["Horizon (days ahead)"], df_hor["MAPE (%)"],
             marker="o", linewidth=2, label="MAPE (%)")
    plt.xlabel("Forecast horizon (days ahead)")
    plt.ylabel("Error")
    plt.title("Hybrid QGA–QPSO: True Horizon Forecast Performance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig_True_Horizon_Performance.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ Saved: Fig_True_Horizon_Performance.png")
    return df_hor


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    DATA_FILE = "Rice_merged_master (1).xlsx"

    # ── Step 1: Descriptive stats ───────────────────────────
    desc_df = compute_descriptive_stats(DATA_FILE)

    # ── Step 2: ACF & Periodogram ──────────────────────────
    df_raw = load_and_clean(DATA_FILE)
    series_list, series_names = build_series(df_raw)
    stationary_list, labels   = zip(*[make_stationary(s) for s in series_list])
    stationary_label           = labels[0]
    plot_acf_figure(list(stationary_list), series_names, stationary_label)
    plot_periodogram(list(stationary_list), series_names, stationary_label)

    # ── Step 3: Feature engineering ────────────────────────
    df_fe = engineer_features(df_raw)

    # ── Step 4: Split ──────────────────────────────────────
    df = pd.read_parquet("X_full.parquet")
    df["DATE_STD"] = pd.to_datetime(df["DATE_STD"])
    df = df.sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)

    trainval_idx, test_idx, split_summary = split_statewise(
        df, test_frac=TEST_FRACTION, min_rows=MIN_ROWS_STATE
    )
    folds, fold_summaries = build_expanding_folds(
        df, trainval_idx, N_FOLDS, VAL_BLOCK_FRAC, MIN_VAL_BLOCK
    )

    Path(".").mkdir(exist_ok=True)
    with open("fold_indices.pkl", "wb") as f:
        pickle.dump({"trainval_idx": trainval_idx, "test_idx": test_idx, "folds": folds}, f)
    print("✅ Saved: fold_indices.pkl")

    # ── Step 5: Internal split for QFS + QPSO ─────────────
    df_tv  = df.loc[trainval_idx].copy().sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)
    df_tr_int, df_va_int = internal_split_statewise(df_tv, val_frac=0.2, min_rows=MIN_ROWS_STATE)

    drop_cols       = ["STATE_KEY", "DATE_STD", "y"]
    feature_cols_full = [c for c in df_tv.columns if c not in drop_cols]

    Xtr_full = df_tr_int[feature_cols_full].to_numpy(dtype=np.float32)
    ytr      = df_tr_int["y"].to_numpy(dtype=np.float32)
    Xva_full = df_va_int[feature_cols_full].to_numpy(dtype=np.float32)
    yva      = df_va_int["y"].to_numpy(dtype=np.float32)

    # ── Step 6: QGA feature selection ─────────────────────
    best_mask_qga,  _ = run_qga(pop=28, gens=18, elite_k=6)
    best_mask_qaco, _ = run_qaco(n_ants=20, iters=12)

    QGA_features  = [feature_cols_full[i] for i in np.where(best_mask_qga  == 1)[0]]
    QACO_features = [feature_cols_full[i] for i in np.where(best_mask_qaco == 1)[0]]
    print(f"QGA selected: {len(QGA_features)} | QACO selected: {len(QACO_features)}")

    for name, feats in [("QGA", QGA_features), ("QACO", QACO_features)]:
        with open(f"features_{name}.txt", "w") as f:
            f.writelines(c + "\n" for c in feats)
    print("✅ Saved: features_QGA.txt, features_QACO.txt")

    # ── Step 7: QPSO hyperparameter optimisation ──────────
    Xtr_qga, _ = get_Xy(df_tr_int, QGA_features)
    Xva_qga, _ = get_Xy(df_va_int, QGA_features)
    best_params = run_qpso(Xtr_qga, ytr, Xva_qga, yva)

    with open("best_xgb_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("✅ Saved: best_xgb_params.json")

    # ── Step 8: Hybrid model training & evaluation ────────
    df_trainval = df.loc[trainval_idx].copy()
    df_test     = df.loc[test_idx].copy()

    n = len(df_trainval)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    df_train_h = df_trainval.iloc[:train_end].copy()
    df_val_h   = df_trainval.iloc[train_end:val_end].copy()
    df_test_h  = df_test.copy()

    xgb_trend, res_model, test_metrics = train_hybrid(
        df_train_h, df_val_h, df_test_h, QGA_features
    )

    # ── Step 9: Multi-horizon evaluation ──────────────────
    df_horizon = evaluate_horizons(df_test_h, QGA_features, xgb_trend, res_model)
    df_horizon.to_csv("horizon_results.csv", index=False)
    print("✅ Pipeline complete.")
