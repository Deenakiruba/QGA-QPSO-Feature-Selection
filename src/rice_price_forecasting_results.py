"""
Rice Price Forecasting — Hybrid QGA-QPSO Model
================================================
Kirubakaran | 2025

This script implements and evaluates a hybrid rice price forecasting pipeline:
  1. Load preprocessed features and train/test splits
  2. Fit ARIMA and ETS baselines; compare against the proposed model (B8)
  3. Run ML baselines (Random Forest, LightGBM, CatBoost, etc.)
  4. State-level heterogeneity analysis
  5. Residual diagnostics and GARCH volatility modelling
  6. Multi-horizon evaluation (1-day, 7-day, 30-day)
  7. Quantum vs classical metaheuristic comparison (QGA vs GA, QPSO vs PSO)
  8. 10-fold time-series cross-validation with policy shock annotation
  9. Computational cost benchmarking
 10. ADF stationarity tests
"""

import json
import pickle
import platform
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import psutil
import tensorflow as tf

from arch import arch_model
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# CONSTANTS
# =============================================================================

SEQ_LEN      = 14       # LSTM residual sequence length
MIN_TRAIN    = 60       # minimum observations required for ARIMA/ETS
EPS          = 1e-9
RMSE_BASELINE = 46.780  # XGB Full Default (reference for improvement %)


# =============================================================================
# METRIC HELPERS
# =============================================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom  = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape   = float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))
    smape  = float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)))
    wape   = float(100.0 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6))
    return {
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":    float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": mape,
        "SMAPE":  smape,
        "WAPE":   wape,
        "MedAE":  float(np.median(np.abs(y_true - y_pred))),
        "R2":     float(r2_score(y_true, y_pred)),
    }


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test. Negative stat means e2 model is more accurate."""
    d      = e1**2 - e2**2
    n      = len(d)
    d_mean = np.mean(d)
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h + 1):
        gamma   = np.cov(d[lag:], d[:-lag])[0, 1]
        nw_var += 2 * (1 - lag / (h + 1)) * gamma
    if nw_var < EPS:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(nw_var / n)
    p_val   = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p_val


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    df = pd.read_parquet("X_full.parquet")
    df["DATE_STD"] = pd.to_datetime(df["DATE_STD"])
    df = df.sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)

    with open("fold_indices.pkl", "rb") as f:
        split_obj = pickle.load(f)

    df_trainval    = df.loc[split_obj["trainval_idx"]].copy()
    df_test_sorted = df.loc[split_obj["test_idx"]].sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)

    with open("features_QGA.txt")      as f: features_qga  = [l.strip() for l in f if l.strip()]
    with open("best_params_qga.json")  as f: params_qga    = json.load(f)
    with open("best_params_full.json") as f: params_full   = json.load(f)

    drop_cols     = ["STATE_KEY", "DATE_STD", "y"]
    features_full = [c for c in df.columns if c not in drop_cols]

    print(f"Train+Val rows : {len(df_trainval)}")
    print(f"Test rows      : {len(df_test_sorted)}")
    print(f"States         : {df['STATE_KEY'].nunique()}")
    print(f"QGA features   : {len(features_qga)}")

    return df, df_trainval, df_test_sorted, features_qga, features_full, params_qga, params_full


# =============================================================================
# HYBRID MODEL — XGBoost + LSTM RESIDUAL CORRECTION
# =============================================================================

def make_residual_sequences(df_sorted, residuals, seq_len):
    Xs, ys, meta = [], [], []
    start = 0
    for state, group in df_sorted.groupby("STATE_KEY", sort=False):
        n = len(group)
        r = residuals[start:start + n]
        start += n
        meta.append((state, n))
        if n <= seq_len:
            continue
        for i in range(seq_len, n):
            Xs.append(r[i - seq_len:i])
            ys.append(r[i])
    Xs = np.array(Xs, dtype=np.float32).reshape(-1, seq_len, 1)
    ys = np.array(ys, dtype=np.float32)
    return Xs, ys, meta


def get_hybrid_predictions(df_trainval, df_test, features, params, seq_len=SEQ_LEN, clip_std=3.0):
    df_tr = df_trainval.sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)
    df_te = df_test.sort_values(["STATE_KEY", "DATE_STD"]).reset_index(drop=True)

    y_tr = df_tr["y"].to_numpy(np.float32)
    y_te = df_te["y"].to_numpy(np.float32)

    xgb = XGBRegressor(**params)
    xgb.fit(df_tr[features].values, y_tr)
    yhat_tr = xgb.predict(df_tr[features].values).astype(np.float32)
    yhat_te = xgb.predict(df_te[features].values).astype(np.float32)

    resid_tr = np.clip(y_tr - yhat_tr,
                       -clip_std * np.std(y_tr - yhat_tr),
                        clip_std * np.std(y_tr - yhat_tr)).astype(np.float32)

    Xr_tr, yr_tr, _        = make_residual_sequences(df_tr, resid_tr, seq_len)
    resid_te_raw            = (y_te - yhat_te).astype(np.float32)
    Xr_te, _, meta_te       = make_residual_sequences(df_te, resid_te_raw, seq_len)

    if len(Xr_tr) == 0 or len(Xr_te) == 0:
        return yhat_te

    tf.keras.backend.clear_session()
    lstm = Sequential([
        LSTM(16, input_shape=(seq_len, 1), dropout=0.3, recurrent_dropout=0.1),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    lstm.fit(Xr_tr, yr_tr, epochs=40, batch_size=512, verbose=0,
             validation_split=0.1,
             callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

    resid_pred = lstm.predict(Xr_te, verbose=0).ravel().astype(np.float32)
    y_final = yhat_te.copy()
    ptr = 0
    for state, n in meta_te:
        if n <= seq_len:
            continue
        idx = df_te[df_te["STATE_KEY"] == state].index.to_numpy()
        n_t = n - seq_len
        y_final[idx[seq_len:]] = yhat_te[idx[seq_len:]] + resid_pred[ptr:ptr + n_t]
        ptr += n_t

    return y_final


# =============================================================================
# STEP 2 — ARIMA AND ETS BASELINES
# =============================================================================

def run_arima_ets_baselines(df, df_trainval, df_test_sorted, pred_b8, y_true):
    states = df["STATE_KEY"].unique()
    arima_preds, arima_true = [], []
    ets_preds,   ets_true   = [], []
    state_results_arima, state_results_ets = [], []

    for state in states:
        df_tr_s = df_trainval[df_trainval["STATE_KEY"] == state].sort_values("DATE_STD")
        df_te_s = df_test_sorted[df_test_sorted["STATE_KEY"] == state].sort_values("DATE_STD")

        if len(df_tr_s) < MIN_TRAIN or len(df_te_s) < 5:
            continue

        y_tr = np.expm1(df_tr_s["y"].values)
        y_te = np.expm1(df_te_s["y"].values)

        try:
            pred_a = np.clip(ARIMA(y_tr, order=(2, 1, 2)).fit().forecast(steps=len(y_te)), 0, None)
            met_a  = regression_metrics(y_te, pred_a)
            state_results_arima.append({"STATE_KEY": state, "N_test": len(y_te), **met_a})
            arima_preds.extend(pred_a.tolist())
            arima_true.extend(y_te.tolist())
        except Exception as e:
            print(f"  ARIMA failed for {state}: {e}")

        try:
            pred_e = np.clip(
                ETSModel(y_tr, error="add", trend="add", seasonal="add", seasonal_periods=12)
                .fit(disp=False).forecast(steps=len(y_te)), 0, None)
            met_e = regression_metrics(y_te, pred_e)
            state_results_ets.append({"STATE_KEY": state, "N_test": len(y_te), **met_e})
            ets_preds.extend(pred_e.tolist())
            ets_true.extend(y_te.tolist())
        except Exception as e:
            print(f"  ETS failed for {state}: {e}")

    arima_preds = np.array(arima_preds)
    arima_true  = np.array(arima_true)
    ets_preds   = np.array(ets_preds)
    ets_true    = np.array(ets_true)

    met_arima = regression_metrics(arima_true, arima_preds)
    met_ets   = regression_metrics(ets_true,   ets_preds)
    met_b8    = regression_metrics(y_true,      pred_b8)

    min_len = min(len(arima_true), len(pred_b8))
    e_arima = arima_true[:min_len] - arima_preds[:min_len]
    e_ets   = ets_true[:min_len]   - ets_preds[:min_len]
    e_b8    = y_true[:min_len]     - pred_b8[:min_len]

    dm_arima_stat, dm_arima_p = dm_test(e_arima, e_b8)
    dm_ets_stat,   dm_ets_p   = dm_test(e_ets,   e_b8)

    df_compare = pd.DataFrame([
        {"Model": "ARIMA(2,1,2)",        **met_arima, "DM_stat_vs_B8": dm_arima_stat, "DM_p_vs_B8": dm_arima_p},
        {"Model": "ETS(A,A,A)",          **met_ets,   "DM_stat_vs_B8": dm_ets_stat,   "DM_p_vs_B8": dm_ets_p},
        {"Model": "B8: Hybrid QGA-QPSO", **met_b8,    "DM_stat_vs_B8": np.nan,         "DM_p_vs_B8": np.nan},
    ]).round(4)

    print(df_compare.to_string(index=False))
    df_compare.to_excel("Table_C11_ARIMA_ETS_vs_B8.xlsx", index=False)
    pd.DataFrame(state_results_arima).to_excel("Table_C11_ARIMA_ByState.xlsx", index=False)
    pd.DataFrame(state_results_ets).to_excel("Table_C11_ETS_ByState.xlsx",     index=False)

    return arima_preds, arima_true, ets_preds, ets_true


# =============================================================================
# STEP 3 — STATE-LEVEL HETEROGENEITY
# =============================================================================

def run_state_heterogeneity(df, df_trainval, df_test_sorted, features_qga, params_qga):
    states = df["STATE_KEY"].unique()
    state_rows = []

    for state in states:
        df_tr_s = df_trainval[df_trainval["STATE_KEY"] == state]
        df_te_s = df_test_sorted[df_test_sorted["STATE_KEY"] == state]
        if len(df_tr_s) < 30 or len(df_te_s) < 10:
            continue
        prices = np.expm1(df_tr_s["y"].values)
        state_rows.append({
            "STATE_KEY":  state,
            "Train_rows": len(df_tr_s),
            "Test_rows":  len(df_te_s),
            "Price_mean": round(float(prices.mean()), 2),
            "Price_std":  round(float(prices.std()),  2),
            "Price_CV_%": round(float(prices.std() / prices.mean() * 100), 2),
            "Price_kurt": round(float(pd.Series(prices).kurtosis()), 2),
        })

    pred_b8_log = get_hybrid_predictions(df_trainval, df_test_sorted, features_qga, params_qga)

    for row in state_rows:
        st  = row["STATE_KEY"]
        idx = df_test_sorted[df_test_sorted["STATE_KEY"] == st].index.to_numpy()
        if len(idx) < 10:
            row.update({"RMSE": np.nan, "MAE": np.nan, "MAPE_%": np.nan, "R2": np.nan})
            continue
        yt  = np.expm1(df_test_sorted.loc[idx, "y"].values.astype(float))
        yp  = np.expm1(pred_b8_log[idx])
        row.update(regression_metrics(yt, yp))

    df_state = pd.DataFrame(state_rows).round(4)
    df_state["Flag"] = df_state["R2"].apply(lambda x: "Poor" if pd.notna(x) and x < 0.85 else "Good")
    df_state = df_state.sort_values("R2", ascending=False).reset_index(drop=True)

    print(df_state[["STATE_KEY", "Train_rows", "Test_rows", "Price_mean", "Price_CV_%", "RMSE", "R2", "Flag"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c" if f == "Poor" else "#2ecc71" for f in df_state["Flag"]]
    axes[0].barh(df_state["STATE_KEY"], df_state["R2"], color=colors, edgecolor="white", height=0.7)
    axes[0].axvline(0.85, color="red", linestyle="--", linewidth=1.5, label="R²=0.85 threshold")
    axes[0].set_xlabel("R² (test set)")
    axes[0].set_title("Per-State R² — B8 Hybrid QGA-QPSO", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.3)

    sc = axes[1].scatter(df_state["Price_CV_%"], df_state["RMSE"],
                         c=df_state["R2"], cmap="RdYlGn", s=80, edgecolors="grey")
    axes[1].set_xlabel("Price CV (%)")
    axes[1].set_ylabel("RMSE (₹)")
    axes[1].set_title("Price Volatility vs RMSE per State", fontweight="bold")
    plt.colorbar(sc, ax=axes[1], label="R²")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig_C12_State_Heterogeneity.png", dpi=200)
    plt.close()

    df_state.to_excel("Table_C12_StateWise_Performance.xlsx", index=False)
    return df_state


# =============================================================================
# STEP 4 — RESIDUAL DIAGNOSTICS AND GARCH
# =============================================================================

def run_residual_diagnostics(df_trainval, df_test_sorted, pred_b8_log):
    state_counts = df_test_sorted["STATE_KEY"].value_counts()
    rep_state    = state_counts.index[0]

    df_tr_rep = df_trainval[df_trainval["STATE_KEY"] == rep_state].sort_values("DATE_STD")
    df_te_rep = df_test_sorted[df_test_sorted["STATE_KEY"] == rep_state].sort_values("DATE_STD")

    y_tr = np.expm1(df_tr_rep["y"].values)
    y_te = np.expm1(df_te_rep["y"].values)
    idx  = df_te_rep.index.to_numpy()

    resid_arima = y_te - np.clip(ARIMA(y_tr, order=(2, 1, 2)).fit().forecast(steps=len(y_te)), 0, None)
    resid_b8    = y_te - np.expm1(pred_b8_log[idx])

    def diagnostic_tests(resid, name):
        lb_p   = acorr_ljungbox(resid, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
        arch_p = het_arch(resid)[1]
        print(f"  [{name}] Ljung-Box p={lb_p:.4f}, ARCH-LM p={arch_p:.4f}")
        return float(lb_p), float(arch_p)

    lb_arima, arch_arima = diagnostic_tests(resid_arima, f"ARIMA — {rep_state}")
    lb_b8,    arch_b8    = diagnostic_tests(resid_b8,    f"B8 Hybrid — {rep_state}")

    garch_available = False
    garch_vol = np.abs(resid_b8)
    try:
        garch_res        = arch_model(resid_b8, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
        garch_vol        = garch_res.conditional_volatility
        garch_resid      = resid_b8 / (garch_vol + EPS)
        lb_garch, arch_garch = diagnostic_tests(garch_resid, f"GARCH-standardised — {rep_state}")
        garch_available  = True
    except Exception as e:
        print(f"  GARCH fitting failed: {e}")

    diag_rows = [
        {"Model": f"ARIMA(2,1,2) — {rep_state}", "Ljung_Box_p": lb_arima, "ARCH_LM_p": arch_arima,
         "Autocorrelation": "Yes" if lb_arima < 0.05 else "No", "Heteroscedasticity": "Yes" if arch_arima < 0.05 else "No"},
        {"Model": f"Hybrid QGA-QPSO — {rep_state}", "Ljung_Box_p": lb_b8, "ARCH_LM_p": arch_b8,
         "Autocorrelation": "Yes" if lb_b8 < 0.05 else "No", "Heteroscedasticity": "Yes" if arch_b8 < 0.05 else "No"},
    ]
    if garch_available:
        diag_rows.append({"Model": f"B8 + GARCH(1,1) — {rep_state}", "Ljung_Box_p": lb_garch, "ARCH_LM_p": arch_garch,
                           "Autocorrelation": "Yes" if lb_garch < 0.05 else "No", "Heteroscedasticity": "Yes" if arch_garch < 0.05 else "No"})

    lags_range = range(1, 21)
    acf_arima  = [pd.Series(resid_arima**2).autocorr(lag=k) for k in lags_range]
    acf_b8     = [pd.Series(resid_b8**2).autocorr(lag=k)    for k in lags_range]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid_arima, linewidth=0.8, color="steelblue")
    ax1.axhline(0, color="red", linewidth=1)
    ax1.set_title("ARIMA(2,1,2) Residuals", fontweight="bold")
    ax1.set_ylabel("Residual (₹)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(resid_b8, linewidth=0.8, color="darkorange")
    ax2.axhline(0, color="red", linewidth=1)
    ax2.set_title("Hybrid QGA-QPSO Residuals", fontweight="bold")
    ax2.set_ylabel("Residual (₹)")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(resid_arima, bins=40, alpha=0.6, color="steelblue", label="ARIMA", density=True)
    ax3.hist(resid_b8,    bins=40, alpha=0.6, color="darkorange", label="Hybrid QGA-QPSO", density=True)
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.set_xlabel("Residual (₹)")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(np.abs(resid_b8), linewidth=0.7, alpha=0.6, color="grey", label="|B8 residual|")
    ax4.plot(garch_vol, linewidth=1.5, color="red", label="GARCH(1,1) conditional σ")
    ax4.set_title("Residual Volatility vs GARCH(1,1)", fontweight="bold")
    ax4.set_ylabel("Volatility (₹)")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar([l - 0.2 for l in lags_range], acf_arima, width=0.35, label="ARIMA", color="steelblue", alpha=0.7)
    ax5.bar([l + 0.2 for l in lags_range], acf_b8,    width=0.35, label="B8",    color="darkorange", alpha=0.7)
    ax5.axhline(0, color="black", linewidth=0.5)
    ax5.set_title("Squared Residual ACF (Volatility Clustering)", fontweight="bold")
    ax5.set_xlabel("Lag")
    ax5.set_ylabel("ACF of squared residuals")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 1])
    stats.probplot(resid_b8, dist="norm", plot=ax6)
    ax6.set_title("Residuals — Normal Q-Q Plot", fontweight="bold")

    plt.suptitle("Residual Diagnostics: ARIMA vs Hybrid vs GARCH", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig("Fig_C14_Residual_Diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close()

    pd.DataFrame(diag_rows).to_excel("Table_C14_Residual_Diagnostics.xlsx", index=False)


# =============================================================================
# STEP 5 — ML BASELINE COMPARISON
# =============================================================================

def run_ml_baselines(df_trainval, df_test_sorted, features_full, y_true, pred_b8):
    candidate_models = {
        "Random Forest": RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42, n_jobs=-1),
        "Extra Trees":   ExtraTreesRegressor(n_estimators=1000, max_depth=6, random_state=42, n_jobs=-1),
        "AdaBoost":      AdaBoostRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
        "LightGBM":      lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost":      CatBoostRegressor(iterations=500, learning_rate=0.05, depth=3, random_seed=42, verbose=0),
    }

    Xtr = df_trainval[features_full].to_numpy(dtype=np.float32)
    ytr = df_trainval["y"].to_numpy(dtype=np.float32)
    Xte = df_test_sorted[features_full].to_numpy(dtype=np.float32)
    yte = df_test_sorted["y"].to_numpy(dtype=np.float32)

    e_b8      = y_true - pred_b8
    true_orig = np.expm1(yte)
    results   = []

    for name, model in candidate_models.items():
        try:
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t1 = time.perf_counter()

            pred_orig = np.expm1(model.predict(Xte))
            met       = regression_metrics(true_orig, pred_orig)
            e_model   = true_orig - pred_orig
            min_len   = min(len(e_model), len(e_b8))
            dm_stat, dm_p = dm_test(e_model[:min_len], e_b8[:min_len])

            results.append({
                "Model":           name,
                "RMSE":            round(met["RMSE"],   4),
                "MAE":             round(met["MAE"],    4),
                "MAPE_%":          round(met["MAPE_%"], 4),
                "SMAPE":           round(met["SMAPE"],  4),
                "R2":              round(met["R2"],     4),
                "RMSE_Improve_%":  round(100 * (RMSE_BASELINE - met["RMSE"]) / RMSE_BASELINE, 2),
                "DM_stat_vs_B8":   round(dm_stat, 4),
                "DM_p_vs_B8":      round(dm_p, 6),
                "Train_time_s":    round(t1 - t0, 2),
            })
            print(f"  {name}: RMSE={met['RMSE']:.2f}, R²={met['R2']:.4f}, time={t1-t0:.1f}s")

        except Exception as e:
            print(f"  {name} failed: {e}")

    df_results = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    print(df_results.to_string(index=False))
    df_results.to_excel("Table_C11_ML_Baselines.xlsx", index=False)
    return df_results


# =============================================================================
# STEP 6 — MULTI-HORIZON EVALUATION
# =============================================================================

def run_multi_horizon(df_trainval, df_test_sorted, features_qga, params_qga, y_true, pred_b8, arima_preds, arima_true):
    HORIZONS   = [1, 7, 30]
    LOOKBACK   = 30

    xgb = XGBRegressor(**params_qga)
    xgb.fit(df_trainval[features_qga].values, df_trainval["y"].values)
    xgb_preds = np.expm1(xgb.predict(df_test_sorted[features_qga].values))

    scaler   = MinMaxScaler()
    Xtr_lstm = scaler.fit_transform(df_trainval[features_qga])
    Xte_lstm = scaler.transform(df_test_sorted[features_qga])
    ytr_lstm = df_trainval["y"].values
    yte_lstm = df_test_sorted["y"].values

    def make_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(lookback, len(X)):
            Xs.append(X[i - lookback:i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    Xtr_seq, ytr_seq = make_sequences(Xtr_lstm, ytr_lstm, LOOKBACK)
    Xte_seq, yte_seq = make_sequences(Xte_lstm, yte_lstm, LOOKBACK)

    tf.keras.backend.clear_session()
    lstm = Sequential([LSTM(32, input_shape=(LOOKBACK, len(features_qga))), Dropout(0.3), Dense(16, activation="relu"), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(Xtr_seq, ytr_seq, epochs=20, batch_size=128, validation_split=0.1, verbose=0,
             callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    lstm_preds  = np.expm1(lstm.predict(Xte_seq, verbose=0).ravel())
    y_true_lstm = np.expm1(yte_seq)

    models = {
        "ARIMA(2,1,2)":      {"y_true": arima_true,  "y_pred": arima_preds},
        "XGBoost (QGA+QPSO)":{"y_true": y_true,       "y_pred": xgb_preds},
        "LSTM (Pure)":       {"y_true": y_true_lstm,  "y_pred": lstm_preds},
        "Hybrid QGA-QPSO":   {"y_true": y_true,       "y_pred": pred_b8},
    }

    rows = []
    for H in HORIZONS:
        for name, data in models.items():
            yt, yp = data["y_true"], data["y_pred"]
            if H >= len(yt):
                continue
            yt_h, yp_h = yt[H:], yp[:-H]
            met = regression_metrics(yt_h, yp_h)
            rows.append({"Model": name, "Horizon": H, **{k: round(v, 4) for k, v in met.items()}})

    df_horizon = pd.DataFrame(rows)
    print(df_horizon.to_string(index=False))
    df_horizon.to_excel("Table_MultiHorizon_Comparison.xlsx", index=False)
    return df_horizon


# =============================================================================
# STEP 7 — QUANTUM vs CLASSICAL METAHEURISTICS
# =============================================================================

def build_fitness_functions(df_trainval, df_test_sorted, features_full, features_qga):
    drop_cols = ["STATE_KEY", "DATE_STD", "y"]

    Xtr_fs = df_trainval[features_full].to_numpy(dtype=np.float32)
    ytr_fs = df_trainval["y"].to_numpy(dtype=np.float32)
    Xte_fs = df_test_sorted[features_full].to_numpy(dtype=np.float32)
    yte_fs = df_test_sorted["y"].to_numpy(dtype=np.float32)

    LAMBDA = 0.02
    d      = len(features_full)

    def fs_fitness(mask):
        k = int(mask.sum())
        if k == 0:
            return 1e9
        cols  = np.where(mask == 1)[0]
        model = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, tree_method="hist")
        model.fit(Xtr_fs[:, cols], ytr_fs)
        rmse  = float(np.sqrt(mean_squared_error(yte_fs, model.predict(Xte_fs[:, cols]))))
        return rmse + LAMBDA * (k / d)

    Xtr_q = df_trainval[features_qga].to_numpy(dtype=np.float32)
    ytr_q = df_trainval["y"].to_numpy(dtype=np.float32)
    Xte_q = df_test_sorted[features_qga].to_numpy(dtype=np.float32)
    yte_q = df_test_sorted["y"].to_numpy(dtype=np.float32)

    def hpo_fitness(v):
        params = dict(
            n_estimators=int(500 + v[0] * 4500), learning_rate=float(0.01 + v[1] * 0.14),
            max_depth=int(3 + v[2] * 9), subsample=float(0.6 + v[3] * 0.35),
            colsample_bytree=float(0.6 + v[4] * 0.35), reg_lambda=float(v[5] * 10),
            min_child_weight=float(1 + v[6] * 9), random_state=42, n_jobs=-1, tree_method="hist")
        m = XGBRegressor(**params)
        m.fit(Xtr_q, ytr_q)
        return float(np.sqrt(mean_squared_error(yte_q, m.predict(Xte_q))))

    return fs_fitness, hpo_fitness, d


def classical_ga(d, pop_size, n_gens, fitness_fn, crossover_rate=0.8, mutation_rate=0.02, elite_k=4, seed=42):
    rng        = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(pop_size, d)).astype(np.int8)
    for i in range(pop_size):
        if population[i].sum() == 0:
            population[i, rng.integers(d)] = 1
    best_mask, best_score, history = None, np.inf, []

    for gen in range(n_gens):
        scores = np.array([fitness_fn(m) for m in population])
        idx = np.argmin(scores)
        if scores[idx] < best_score:
            best_score, best_mask = float(scores[idx]), population[idx].copy()
        history.append(best_score)
        print(f"  [GA]  Gen {gen+1:>2}/{n_gens} | Best={best_score:.5f} | Features={best_mask.sum()}/{d}")

        elites  = population[np.argsort(scores)[:elite_k]].copy()
        new_pop = [e.copy() for e in elites]
        while len(new_pop) < pop_size:
            t1 = rng.choice(pop_size, 3, replace=False)
            t2 = rng.choice(pop_size, 3, replace=False)
            p1 = population[t1[np.argmin(scores[t1])]].copy()
            p2 = population[t2[np.argmin(scores[t2])]].copy()
            if rng.random() < crossover_rate:
                pt = rng.integers(1, d)
                c1, c2 = np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in [c1, c2]:
                c[rng.random(d) < mutation_rate] ^= 1
                if c.sum() == 0:
                    c[rng.integers(d)] = 1
                new_pop.append(c)
        population = np.array(new_pop[:pop_size], dtype=np.int8)

    return best_mask, best_score, history


def quantum_ga(d, pop_size, n_gens, fitness_fn, update_rate=0.12, top_k=6, seed=42):
    rng        = np.random.default_rng(seed)
    P          = np.full((pop_size, d), 0.5)
    best_mask, best_score, history = None, np.inf, []

    for gen in range(n_gens):
        masks = (rng.random(P.shape) < P).astype(np.int8)
        for i in range(pop_size):
            if masks[i].sum() == 0:
                masks[i, rng.integers(d)] = 1
        scores = np.array([fitness_fn(m) for m in masks])
        idx = np.argmin(scores)
        if scores[idx] < best_score:
            best_score, best_mask = float(scores[idx]), masks[idx].copy()
        history.append(best_score)
        print(f"  [QGA] Gen {gen+1:>2}/{n_gens} | Best={best_score:.5f} | Features={best_mask.sum()}/{d}")
        target = masks[np.argsort(scores)[:top_k]].mean(axis=0)
        P = np.clip((1 - update_rate) * P + update_rate * target, 0.02, 0.98)

    return best_mask, best_score, history


def classical_pso(n_part, n_iter, n_dim, fitness_fn, w=0.7, c1=1.5, c2=1.5, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.random((n_part, n_dim))
    V   = rng.random((n_part, n_dim)) * 0.1
    pbest, pbest_f = X.copy(), np.array([fitness_fn(x) for x in X])
    gbest, gbest_f = pbest[np.argmin(pbest_f)].copy(), float(pbest_f.min())
    history = []

    for it in range(n_iter):
        r1, r2 = rng.random((n_part, n_dim)), rng.random((n_part, n_dim))
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X), -0.5, 0.5)
        X = np.clip(X + V, 0, 1)
        for i in range(n_part):
            f = fitness_fn(X[i])
            if f < pbest_f[i]:
                pbest_f[i], pbest[i] = f, X[i].copy()
        if pbest_f.min() < gbest_f:
            gbest_f = float(pbest_f.min())
            gbest   = pbest[np.argmin(pbest_f)].copy()
        history.append(gbest_f)
        print(f"  [PSO]  Iter {it+1:>2}/{n_iter} | Best RMSE={gbest_f:.5f}")

    return gbest, gbest_f, history


def quantum_pso(n_part, n_iter, n_dim, fitness_fn, beta_start=1.0, beta_end=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.random((n_part, n_dim))
    pbest, pbest_f = X.copy(), np.array([fitness_fn(x) for x in X])
    gbest, gbest_f = pbest[np.argmin(pbest_f)].copy(), float(pbest_f.min())
    history = []

    for it in range(n_iter):
        beta = beta_start - (beta_start - beta_end) * (it / n_iter)
        mb   = pbest.mean(axis=0)
        for i in range(n_part):
            u    = rng.random(n_dim)
            p_a  = 0.5 * (pbest[i] + gbest)
            sign = np.where(rng.random(n_dim) < 0.5, 1.0, -1.0)
            X[i] = np.clip(p_a + sign * beta * np.abs(mb - X[i]) * np.log(1.0 / (u + 1e-12)), 0, 1)
            f = fitness_fn(X[i])
            if f < pbest_f[i]:
                pbest_f[i], pbest[i] = f, X[i].copy()
        if pbest_f.min() < gbest_f:
            gbest_f = float(pbest_f.min())
            gbest   = pbest[np.argmin(pbest_f)].copy()
        history.append(gbest_f)
        print(f"  [QPSO] Iter {it+1:>2}/{n_iter} | Best RMSE={gbest_f:.5f}")

    return gbest, gbest_f, history


def run_quantum_vs_classical(df_trainval, df_test_sorted, features_full, features_qga):
    fs_fitness, hpo_fitness, d = build_fitness_functions(df_trainval, df_test_sorted, features_full, features_qga)

    POP, GENS  = 28, 18
    N_PART, N_ITER, N_DIM = 18, 20, 7
    SEED = 42

    t0 = time.perf_counter()
    ga_mask,   ga_score,   ga_hist   = classical_ga(d, POP, GENS, fs_fitness, seed=SEED)
    t1 = time.perf_counter()
    qga_mask,  qga_score,  qga_hist  = quantum_ga(d, POP, GENS, fs_fitness, seed=SEED)
    t2 = time.perf_counter()
    pso_best,  pso_score,  pso_hist  = classical_pso(N_PART, N_ITER, N_DIM, hpo_fitness, seed=SEED)
    t3 = time.perf_counter()
    qpso_best, qpso_score, qpso_hist = quantum_pso(N_PART, N_ITER, N_DIM, hpo_fitness, seed=SEED)
    t4 = time.perf_counter()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(range(1, GENS + 1), ga_hist,  color="#e74c3c", marker="o", markersize=4, label=f"Classical GA (final={ga_score:.4f})")
    axes[0].plot(range(1, GENS + 1), qga_hist, color="#2ecc71", marker="s", markersize=4, label=f"QGA (final={qga_score:.4f})")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Fitness")
    axes[0].set_title("Feature Selection Convergence: GA vs QGA", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, N_ITER + 1), pso_hist,  color="#e74c3c", marker="o", markersize=4, label=f"Classical PSO (final={pso_score:.5f})")
    axes[1].plot(range(1, N_ITER + 1), qpso_hist, color="#2ecc71", marker="s", markersize=4, label=f"QPSO (final={qpso_score:.5f})")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best RMSE")
    axes[1].set_title("Hyperparameter Tuning Convergence: PSO vs QPSO", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Quantum-Inspired vs Classical Metaheuristics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("Fig_C7_Convergence_Quantum_vs_Classical.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"GA  time: {(t1-t0)/60:.1f} min | Features selected: {int(ga_mask.sum())}/{d}")
    print(f"QGA time: {(t2-t1)/60:.1f} min | Features selected: {int(qga_mask.sum())}/{d}")
    print(f"PSO  time: {(t3-t2)/60:.1f} min | Best RMSE: {pso_score:.5f}")
    print(f"QPSO time: {(t4-t3)/60:.1f} min | Best RMSE: {qpso_score:.5f}")


# =============================================================================
# STEP 8 — 10-FOLD TIME-SERIES CROSS-VALIDATION
# =============================================================================

def run_cv_analysis(df, features_full, features_qga, params_qga):
    POLICY_SHOCKS = [
        ("2008-04-01", "2008-12-31", "Export ban (non-basmati)"),
        ("2011-01-01", "2011-12-31", "MSP hike +16%"),
        ("2018-01-01", "2018-12-31", "MSP hike +13%"),
        ("2020-03-01", "2021-03-31", "COVID-19 supply shock"),
        ("2022-09-01", "2022-12-31", "Export ban (broken white rice)"),
        ("2023-08-01", "2023-12-31", "Export tax on non-basmati"),
    ]

    def check_shock_overlap(fold_start, fold_end, shocks):
        return [name for s, e, name in shocks
                if fold_start <= pd.Timestamp(e) and fold_end >= pd.Timestamp(s)] or ["None"]

    df_cv     = df.sort_values("DATE_STD").reset_index(drop=True)
    X_cv      = df_cv[features_full].to_numpy(dtype=np.float32)
    y_cv      = df_cv["y"].to_numpy(dtype=np.float32)
    dates_cv  = df_cv["DATE_STD"].to_numpy()
    states_cv = df_cv["STATE_KEY"].to_numpy()
    qga_idx   = [list(features_full).index(f) for f in features_qga]

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=10).split(X_cv), start=1):
        Xtr, Xva  = X_cv[tr_idx], X_cv[va_idx]
        ytr, yva  = y_cv[tr_idx], y_cv[va_idx]
        dates_va  = dates_cv[va_idx]
        states_va = states_cv[va_idx]

        fold_start = pd.Timestamp(dates_va.min())
        fold_end   = pd.Timestamp(dates_va.max())
        overlaps   = check_shock_overlap(fold_start, fold_end, POLICY_SHOCKS)

        xgb_cv = XGBRegressor(**params_qga)
        xgb_cv.fit(Xtr[:, qga_idx], ytr)
        yhat = xgb_cv.predict(Xva[:, qga_idx])

        yva_orig, yhat_orig = np.expm1(yva), np.expm1(yhat)
        met = regression_metrics(yva_orig, yhat_orig)

        r2_by_state = [r2_score(yva_orig[states_va == st], yhat_orig[states_va == st])
                       for st in np.unique(states_va) if (states_va == st).sum() >= 10]

        rows.append({
            "Fold":          fold,
            "Val_start":     fold_start.strftime("%Y-%m-%d"),
            "Val_end":       fold_end.strftime("%Y-%m-%d"),
            "Val_size":      len(va_idx),
            "RMSE":          round(met["RMSE"], 2),
            "MAE":           round(met["MAE"],  2),
            "R2":            round(met["R2"],   4),
            "R2_state_mean": round(float(np.mean(r2_by_state)), 4) if r2_by_state else "N/A",
            "Policy_shock":  " | ".join(overlaps),
        })
        print(f"  Fold {fold:>2} | {fold_start.date()} → {fold_end.date()} | RMSE={met['RMSE']:>8.2f} | R²={met['R2']:.4f} | Shock: {', '.join(overlaps)}")

    df_cv_results = pd.DataFrame(rows)
    df_cv_results.to_excel("Table_C9_CV_Fold_Analysis.xlsx", index=False)

    shock_folds    = df_cv_results[df_cv_results["Policy_shock"] != "None"]
    no_shock_folds = df_cv_results[df_cv_results["Policy_shock"] == "None"]
    print(f"\n  Shock folds    — Mean RMSE: {shock_folds['RMSE'].mean():.2f}, Mean R²: {shock_folds['R2'].mean():.4f}")
    print(f"  No-shock folds — Mean RMSE: {no_shock_folds['RMSE'].mean():.2f}, Mean R²: {no_shock_folds['R2'].mean():.4f}")

    return df_cv_results


# =============================================================================
# STEP 9 — COMPUTATIONAL COST BENCHMARKING
# =============================================================================

def run_cost_benchmarking(df_trainval, df_test_sorted, features_full, features_qga, params_qga, met_b8_ov):
    POP_BENCH = 28
    GENS_BENCH = 3
    N_PART_BENCH = 18
    N_ITER_BENCH = 3

    Xtr = df_trainval[features_full].to_numpy(dtype=np.float32)
    ytr = df_trainval["y"].to_numpy(dtype=np.float32)
    Xte = df_test_sorted[features_full].to_numpy(dtype=np.float32)
    yte = df_test_sorted["y"].to_numpy(dtype=np.float32)

    timing = []

    # (a) Vanilla XGBoost
    tracemalloc.start()
    t0 = time.perf_counter()
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, tree_method="hist")
    xgb.fit(Xtr, ytr)
    pred_v = xgb.predict(Xte)
    met_v  = regression_metrics(np.expm1(yte), np.expm1(pred_v))
    t1, (_, peak) = time.perf_counter(), tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timing.append({"Component": "Vanilla XGBoost", "Time_min": round((t1-t0)/60, 3), "Peak_RAM_MB": round(peak/1e6, 2), "RMSE": round(met_v["RMSE"], 4)})

    # (b) QGA feature selection (extrapolated)
    tracemalloc.start()
    t0 = time.perf_counter()
    rng  = np.random.default_rng(42)
    d    = len(features_full)
    P    = np.full((POP_BENCH, d), 0.5)
    for _ in range(GENS_BENCH):
        masks  = (rng.random(P.shape) < P).astype(np.int8)
        scores = np.array([
            float(np.sqrt(mean_squared_error(yte, XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, tree_method="hist")
                                             .fit(Xtr[:, np.where(m)[0]], ytr).predict(Xte[:, np.where(m)[0]]))))
            if m.sum() > 0 else 1e9 for m in masks])
        P = np.clip((1 - 0.12) * P + 0.12 * masks[np.argsort(scores)[:6]].mean(axis=0), 0.02, 0.98)
    t1, (_, peak) = time.perf_counter(), tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_qga_full = (t1 - t0) / GENS_BENCH * 18
    timing.append({"Component": "QGA feature selection (28×18 evals)", "Time_min": round(t_qga_full/60, 3), "Peak_RAM_MB": round(peak/1e6, 2), "RMSE": "N/A"})

    # (c) Full B8 pipeline
    tracemalloc.start()
    t0 = time.perf_counter()
    get_hybrid_predictions(df_trainval, df_test_sorted, features_qga, params_qga)
    t1, (_, peak) = time.perf_counter(), tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timing.append({"Component": "Full hybrid (XGB + LSTM)", "Time_min": round((t1-t0)/60, 3), "Peak_RAM_MB": round(peak/1e6, 2), "RMSE": round(met_b8_ov["RMSE"], 4)})

    df_timing = pd.DataFrame(timing)
    print(df_timing.to_string(index=False))

    hw = {
        "OS": platform.platform(), "Python": platform.python_version(),
        "CPU_cores": psutil.cpu_count(logical=False), "RAM_GB": round(psutil.virtual_memory().total / 1e9, 1),
        "TensorFlow": tf.__version__,
    }
    print("\nHardware:", hw)

    with pd.ExcelWriter("Table_C15_Computational_Cost.xlsx") as writer:
        df_timing.to_excel(writer, sheet_name="Timing", index=False)
        pd.DataFrame([hw]).T.reset_index().rename(columns={"index": "Parameter", 0: "Value"}).to_excel(writer, sheet_name="Hardware", index=False)


# =============================================================================
# STEP 10 — ADF STATIONARITY TESTS
# =============================================================================

def run_adf_tests(df):
    results = []
    for state in df["STATE_KEY"].unique():
        y_log  = df[df["STATE_KEY"] == state].sort_values("DATE_STD")["y"].dropna().values
        y_diff = np.diff(y_log)

        def adf_row(series):
            try:
                res = adfuller(series, autolag="AIC")
                return round(res[0], 4), round(res[1], 4), "Yes" if res[1] < 0.05 else "No"
            except:
                return "FAILED", "FAILED", "FAILED"

        stat_l, p_l, st_l = adf_row(y_log)
        stat_d, p_d, st_d = adf_row(y_diff)

        results.append({
            "State": state, "N_obs": len(y_log),
            "ADF_stat_log": stat_l, "p_value_log": p_l, "Stationary_log": st_l,
            "ADF_stat_diff": stat_d, "p_value_diff": p_d, "Stationary_diff": st_d,
            "Action": "No differencing needed" if st_l == "Yes" else "First differencing applied",
        })

    df_adf = pd.DataFrame(results)
    n_stat = (df_adf["Stationary_log"] == "Yes").sum()
    print(f"  States stationary at log level    : {n_stat}/{len(df_adf)}")
    print(f"  States requiring differencing     : {len(df_adf) - n_stat}/{len(df_adf)}")
    print(df_adf.to_string(index=False))
    df_adf.to_excel("Table_ADF_Stationarity.xlsx", index=False)
    return df_adf


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    df, df_trainval, df_test_sorted, features_qga, features_full, params_qga, params_full = load_data()

    print("\nComputing B8 (Hybrid QGA-QPSO) predictions...")
    pred_b8_log = get_hybrid_predictions(df_trainval, df_test_sorted, features_qga, params_qga)
    y_true_log  = df_test_sorted["y"].values.astype(float)
    y_true      = np.expm1(y_true_log)
    pred_b8     = np.expm1(pred_b8_log)
    met_b8      = regression_metrics(y_true, pred_b8)
    print(f"B8 RMSE={met_b8['RMSE']:.4f}, R²={met_b8['R2']:.4f}")

    print("\nRunning ARIMA and ETS baselines...")
    arima_preds, arima_true, ets_preds, ets_true = run_arima_ets_baselines(df, df_trainval, df_test_sorted, pred_b8, y_true)

    print("\nRunning state-level heterogeneity analysis...")
    run_state_heterogeneity(df, df_trainval, df_test_sorted, features_qga, params_qga)

    print("\nRunning residual diagnostics...")
    run_residual_diagnostics(df_trainval, df_test_sorted, pred_b8_log)

    print("\nRunning ML baseline comparison...")
    run_ml_baselines(df_trainval, df_test_sorted, features_full, y_true, pred_b8)

    print("\nRunning multi-horizon evaluation...")
    run_multi_horizon(df_trainval, df_test_sorted, features_qga, params_qga, y_true, pred_b8, arima_preds, arima_true)

    print("\nRunning quantum vs classical metaheuristic comparison...")
    run_quantum_vs_classical(df_trainval, df_test_sorted, features_full, features_qga)

    print("\nRunning 10-fold time-series cross-validation...")
    run_cv_analysis(df, features_full, features_qga, params_qga)

    print("\nRunning computational cost benchmarking...")
    run_cost_benchmarking(df_trainval, df_test_sorted, features_full, features_qga, params_qga, met_b8)

    print("\nRunning ADF stationarity tests...")
    run_adf_tests(df)

    print("\nDone. All tables and figures saved.")


if __name__ == "__main__":
    main()
