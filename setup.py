from setuptools import setup, find_packages

setup(
    name="rice_price_forecasting",
    version="1.0.0",
    description="Hybrid QGA–QPSO XGBoost–LSTM pipeline for rice price forecasting",
    author="Lourdeena J S",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "tensorflow>=2.13",
        "statsmodels>=0.14",
        "matplotlib>=3.7",
        "openpyxl>=3.1",
        "pyarrow>=13.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
