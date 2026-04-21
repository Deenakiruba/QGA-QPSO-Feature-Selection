"""
Unit tests for the Rice Price Forecasting pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rice_price_forecasting import (
    mape_safe,
    smape,
    wape,
    eval_metrics,
    make_seq,
    make_seq_X,
    qga_init,
    qga_measure,
    qga_update,
    vec_to_xgb_params,
    engineer_features,
)


# -----------------------------------------------------------
# Metric tests
# -----------------------------------------------------------

def test_mape_safe_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert mape_safe(y, y) == pytest.approx(0.0)


def test_mape_safe_zero_denom():
    """Should not raise ZeroDivisionError when y_true has zeros."""
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([1.0, 100.0])
    result = mape_safe(y_true, y_pred)
    assert np.isfinite(result)


def test_smape_symmetric():
    """sMAPE of (a,b) == sMAPE of (b,a)."""
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 140.0, 210.0])
    assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))


def test_wape_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert wape(y, y) == pytest.approx(0.0)


def test_eval_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = eval_metrics(y, y)
    assert set(m.keys()) == {"RMSE", "MAE", "MAPE_%", "SMAPE_%", "WAPE_%", "MedAE", "R2"}


def test_eval_metrics_perfect_r2():
    y = np.arange(1.0, 11.0)
    m = eval_metrics(y, y)
    assert m["R2"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(0.0)


# -----------------------------------------------------------
# Sequence creation tests
# -----------------------------------------------------------

def test_make_seq_shape():
    X = np.random.rand(50, 5).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)
    Xs, ys = make_seq(X, y, seq_len=7)
    assert Xs.shape == (43, 7, 5)   # 50 - 7 = 43
    assert ys.shape == (43,)


def test_make_seq_X_shape():
    X = np.random.rand(30, 4).astype(np.float32)
    Xs = make_seq_X(X, seq_len=5)
    assert Xs.shape == (25, 5, 4)   # 30 - 5 = 25


def test_make_seq_no_lookahead():
    """Target at position i must only use features up to position i."""
    n, d, seq = 20, 3, 5
    X = np.arange(n * d, dtype=np.float32).reshape(n, d)
    y = np.arange(n, dtype=np.float32)
    Xs, ys = make_seq(X, y, seq_len=seq)
    # First target should correspond to index seq
    assert ys[0] == pytest.approx(y[seq])
    # First sequence window should be rows 0..seq-1
    np.testing.assert_array_equal(Xs[0], X[:seq])


# -----------------------------------------------------------
# QGA tests
# -----------------------------------------------------------

def test_qga_init_shape():
    P = qga_init(pop=10, d=20)
    assert P.shape == (10, 20)


def test_qga_init_values():
    P = qga_init(pop=5, d=15, p0=0.5)
    assert np.allclose(P, 0.5)


def test_qga_measure_binary():
    rng = np.random.default_rng(0)
    P = np.full((8, 10), 0.5)
    masks = qga_measure(P, rng)
    assert set(np.unique(masks)).issubset({0, 1})
    assert masks.shape == P.shape


def test_qga_update_clipped():
    rng = np.random.default_rng(0)
    P = np.full((4, 6), 0.5)
    masks = qga_measure(P, rng)
    P_new = qga_update(P, masks, lr=0.12)
    assert P_new.min() >= 0.02 - 1e-9
    assert P_new.max() <= 0.98 + 1e-9


# -----------------------------------------------------------
# QPSO parameter mapping tests
# -----------------------------------------------------------

def test_vec_to_xgb_params_bounds():
    v_low  = np.zeros(7)
    v_high = np.ones(7)
    p_low  = vec_to_xgb_params(v_low)
    p_high = vec_to_xgb_params(v_high)

    assert p_low["learning_rate"]  == pytest.approx(0.01)
    assert p_high["learning_rate"] == pytest.approx(0.15)
    assert p_low["max_depth"]      == 3
    assert p_high["max_depth"]     == 12
    assert p_low["n_estimators"]   == 500
    assert p_high["n_estimators"]  == 5000


def test_vec_to_xgb_params_types():
    v = np.random.rand(7)
    p = vec_to_xgb_params(v)
    assert isinstance(p["max_depth"],      int)
    assert isinstance(p["n_estimators"],   int)
    assert isinstance(p["learning_rate"],  float)


# -----------------------------------------------------------
# Feature engineering smoke test
# -----------------------------------------------------------

def test_engineer_features_smoke(tmp_path):
    """Create a minimal DataFrame and verify feature engineering runs."""
    rng2 = np.random.default_rng(99)
    n = 200
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "STATE_KEY":        ["TestState"] * n,
        "DATE_STD":         dates,
        "Modal Price":      1500 + rng2.normal(0, 50, n),
        "Min Price":        1450 + rng2.normal(0, 40, n),
        "Max Price":        1550 + rng2.normal(0, 40, n),
        "Price_Change_Percent": rng2.normal(0, 1, n),
        "Latitude":         [13.0] * n,
        "Longitude":        [80.0] * n,
        "T2M_MAX":          28 + rng2.normal(0, 2, n),
        "T2M_MIN":          20 + rng2.normal(0, 2, n),
        "T2M_RANGE":        rng2.uniform(5, 10, n),
        "PRECTOTCORR":      rng2.exponential(2, n),
        "RH2M":             rng2.uniform(50, 80, n),
        "WS2M":             rng2.uniform(2, 5, n),
        "ALLSKY_SFC_SW_DWN": rng2.uniform(14, 22, n),
        "PSC":              rng2.uniform(50, 90, n),
        "PRODUCTION_TONNES": [8e6] * n,
        "AREA_HA":          [2e6] * n,
        "YIELD_TON_PER_HA": [4.0] * n,
    })

    import os
    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        df_out = engineer_features(df)
    finally:
        os.chdir(orig)

    assert len(df_out) > 0
    assert "y" in df_out.columns
    assert "price_lag_1" in df_out.columns
    assert "GDD" in df_out.columns
    assert "month_sin" in df_out.columns
    assert df_out["y"].isna().sum() == 0
