import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

states = {
    "Tamil Nadu":    (1700, 120),
    "Karnataka":     (1550, 110),
    "Punjab":        (1900, 130),
    "Andhra Pradesh":(1650, 115),
    "West Bengal":   (1480, 100),
}

rows = []
for state, (base_price, price_std) in states.items():
    dates = pd.date_range("2015-01-01", "2022-12-31", freq="W")
    n = len(dates)

    # Simulate a price series with seasonal component
    t = np.arange(n)
    seasonal = 80 * np.sin(2 * np.pi * t / 52) + 40 * np.sin(4 * np.pi * t / 52)
    trend    = 0.8 * t
    noise    = rng.normal(0, price_std * 0.3, n)
    modal_p  = base_price + trend + seasonal + noise
    modal_p  = np.clip(modal_p, 800, 4000)

    lat = rng.uniform(10, 30)
    lon = rng.uniform(74, 90)

    for i, d in enumerate(dates):
        t2m_max = 28 + 8 * np.sin(2 * np.pi * d.dayofyear / 365) + rng.normal(0, 1.5)
        t2m_min = 18 + 6 * np.sin(2 * np.pi * d.dayofyear / 365) + rng.normal(0, 1.2)
        rows.append({
            "STATE_KEY":        state,
            "DATE_STD":         d.strftime("%d/%m/%Y"),
            "Min Price":        round(modal_p[i] * rng.uniform(0.92, 0.97), 2),
            "Max Price":        round(modal_p[i] * rng.uniform(1.03, 1.08), 2),
            "Modal Price":      round(modal_p[i], 2),
            "Price_Change_Percent": round(rng.normal(0, 1.5), 4),
            "Latitude":         round(lat, 4),
            "Longitude":        round(lon, 4),
            "ALLSKY_SFC_SW_DWN": round(rng.uniform(12, 24), 3),
            "T2M_RANGE":        round(t2m_max - t2m_min, 3),
            "T2M_MAX":          round(t2m_max, 3),
            "T2M_MIN":          round(t2m_min, 3),
            "PRECTOTCORR":      round(max(0, rng.exponential(2.5)), 3),
            "RH2M":             round(rng.uniform(45, 85), 3),
            "WS2M":             round(rng.uniform(1.5, 5.5), 3),
            "PSC":              round(rng.uniform(40, 90), 3),
            "PRODUCTION_TONNES": round(rng.uniform(5e6, 15e6), 0),
            "AREA_HA":          round(rng.uniform(1e6, 4e6), 0),
            "YIELD_TON_PER_HA": round(rng.uniform(2.5, 4.5), 3),
        })

df = pd.DataFrame(rows).sample(500, random_state=42).reset_index(drop=True)
out = Path(__file__).parent / "sample_data.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows → {out}")
