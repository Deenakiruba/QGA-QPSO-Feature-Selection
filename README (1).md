# Rice Price Forecasting — Hybrid QGA–QPSO Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete code and data for the paper:
*"Hybrid QGA QPSO Optimised XGBoost LSTM Model for Multi-Horizon Rice Price Forecasting in India"*

---

## What This Does

The pipeline forecasts rice prices across Indian states using a hybrid model that combines:

- **Quantum Genetic Algorithm (QGA)** — selects the most relevant features from a pool of 43
- **Quantum PSO (QPSO)** — tunes XGBoost hyperparameters
- **XGBoost** — models the price trend
- **LSTM** — corrects residual errors from XGBoost

The final model generates 7-day, 14-day, and 30-day ahead price forecasts.

---

## Repository Structure

```
rice_price_forecasting/
├── src/
│   └── rice_price_forecasting.py   ← Main pipeline
├── configs/
│   ├── features_QGA.txt            ← QGA-selected features
│   └── pipeline_config.yaml        ← All tunable parameters
├── data/
│   └── sample/
│       ├── sample_data.csv         ← Sample dataset (500 rows)
│       └── DATA_FORMAT.md          ← Column descriptions
├── notebooks/
│   └── 01_demo_pipeline.ipynb      ← Interactive walkthrough
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## Getting Started

**Using Conda (recommended):**

```bash
git clone https://github.com/<your-username>/rice_price_forecasting.git
cd rice_price_forecasting
conda env create -f environment.yml
conda activate rice-forecast
python src/rice_price_forecasting.py
```

**Using pip:**

```bash
pip install -r requirements.txt
python src/rice_price_forecasting.py
```

---

## Data

Place your Excel file at the project root and set the path in `configs/pipeline_config.yaml`:

```yaml
data:
  file: "Rice_merged_master.xlsx"
```

The file should contain these columns:

| Column | Description |
|--------|-------------|
| `STATE_KEY` | State identifier |
| `DATE_STD` | Date (DD/MM/YYYY) |
| `Modal Price` | Modal market price (₹/quintal) |
| `Min Price`, `Max Price` | Price range |
| `T2M_MAX`, `T2M_MIN` | Temperature (°C) from NASA POWER |
| `PRECTOTCORR` | Precipitation (mm/day) |
| `RH2M` | Relative humidity (%) |
| `WS2M` | Wind speed at 2m (m/s) |
| `ALLSKY_SFC_SW_DWN` | Solar radiation (MJ/m²/day) |
| `PRODUCTION_TONNES`, `AREA_HA`, `YIELD_TON_PER_HA` | Production statistics |

A 500-row sample is available in `data/sample/sample_data.csv` if you want to test the pipeline without the full dataset:

```bash
python src/rice_price_forecasting.py --sample
```

---

## Key Parameters

All parameters are in `configs/pipeline_config.yaml`. The defaults used in the paper are:

```yaml
qga:
  population: 28
  generations: 18
  lambda_feat: 0.02

qpso:
  n_particles: 20
  iterations: 30

hybrid:
  seq_len: 14
  horizons: [7, 14, 30]
  epochs: 30
  batch_size: 256
```

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Citation

If you use this code in your work, please cite:

```bibtex
@article{lourdeena2025rice,
  title   = {Hybrid QGA--QPSO Optimised XGBoost--LSTM Model for
             Multi-Horizon Rice Price Forecasting in India},
  author  = {Lourdeena J S, Graceline Jasmine S and Febin Daya J L},
  journal = {Journal Name},
  year    = {2025},
  doi     = {10.XXXX/XXXXXX}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

For questions, open a GitHub Issue or contact the corresponding author.
