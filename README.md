# Machine Learning-Based Remaining Useful Life Estimation for Turbofan Engines

An undergraduate Machine Learning course project focused on predictive maintenance and Remaining Useful Life (RUL) estimation for aircraft turbofan engines using the benchmark **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset (FD001)**.

---

## 📌 Project Overview & Objective

In aviation and industrial maintenance, unexpected mechanical failures pose severe safety risks and lead to substantial operational downtime. **Predictive Maintenance (PdM)** aims to monitor asset health through continuous sensor telemetry and estimate the **Remaining Useful Life (RUL)**—the number of operational flight cycles remaining before the equipment degrades to failure.

### Key Objectives:
- Ingest and rigorously inspect the NASA C-MAPSS FD001 dataset.
- Perform an exploratory data analysis (EDA) to discover sensor degradation signatures, variance characteristics, and lifetime statistics.
- Build a leak-free preprocessing and feature engineering pipeline (rolling statistics, difference features, piecewise linear RUL targets).
- Train, evaluate, and benchmark regression algorithms:
  1. **Linear Regression** (Baseline model)
  2. **Random Forest Regressor** (Non-linear ensemble model)
  3. **XGBoost Regressor** (High-performance gradient boosted decision trees)
- Evaluate performance using industry-standard metrics (**MAE**, **RMSE**, and **$R^2$**), alongside prognostic criteria.
- Deploy an interactive **Streamlit** dashboard for real-time engine health monitoring.

---

## 📂 Project Directory Structure & Folder Responsibilities

```
turbofan-rul-ml/
│
├── data/
│   ├── raw/                 # Immutable original data files from NASA C-MAPSS
│   │   ├── train_FD001.txt  # Run-to-failure trajectories for 100 training engines
│   │   ├── test_FD001.txt   # Censored trajectories for 100 test engines
│   │   └── RUL_FD001.txt    # Ground truth remaining useful life for test engines
│   └── processed/           # Cleaned, engineered, and train/val split datasets
│
├── notebooks/               # Step-by-step modular Jupyter Notebooks for exploration and experimentation
│   ├── 01_data_understanding.ipynb   # Step 1: Ingestion, schema validation, missing/duplicate audits
│   ├── 02_eda.ipynb                  # Step 2: In-depth ML exploratory data analysis & degradation trends
│   ├── 03_preprocessing.ipynb        # Step 3: Leak-free grouping, splitting, and scaling
│   ├── 04_feature_engineering.ipynb   # Step 4: Rolling statistics, domain sensor features
│   ├── 05_model_training.ipynb       # Step 5: Model fitting and hyperparameter tuning
│   ├── 06_model_comparison.ipynb     # Step 6: Multi-model evaluation across metrics
│   └── 07_test_evaluation.ipynb      # Step 7: Unseen test set inference and error analysis
│
├── src/                     # Production-quality, reusable Python modules
│   ├── __init__.py          # Package initializer
│   ├── data_loader.py       # Robust space-separated data parsing & schema validation
│   ├── preprocessing.py     # Data cleaning, unit-based train/validation splitting
│   ├── feature_engineering.py # Rolling stats, sensor selection, piecewise target clipping
│   ├── train.py             # Script for fitting and serializing regression models
│   ├── evaluate.py          # Metric computation (RMSE, MAE, R², NASA score)
│   └── predict.py           # Inference pipeline for new engine trajectories
│
├── models/                  # Serialized trained model artifacts (.pkl, metadata)
│
├── reports/                 # Project documentation, findings, and technical reports
│   └── figures/             # High-resolution saved visualization charts and plots
│
├── app/                     # Interactive user application
│   └── app.py               # Streamlit interactive dashboard for real-time RUL prediction
│
├── requirements.txt         # Project dependencies and package versions
├── README.md                # Comprehensive project documentation
└── .gitignore               # Ignored cache, virtualenv, and temporary files
```

---

## 📊 Dataset Description (NASA C-MAPSS FD001)

The FD001 dataset represents turbofan degradation simulated under standard sea-level operating conditions with a single failure mode (High-Pressure Compressor degradation):
- **100 Training Engines**: Recorded from nominal health condition until functional failure ($RUL = 0$).
- **100 Test Engines**: Operational profiles ending at an arbitrary time cycle prior to failure.
- **26 Telemetry Channels per Record**:
  - `unit_number`: Engine identifier (integer ID)
  - `time_cycles`: Discrete flight/operating cycle index
  - `operational_setting_1`, `operational_setting_2`, `operational_setting_3`: Ambient flight conditions
  - `sensor_1` through `sensor_21`: Thermodynamic and physical sensor readings (temperatures, pressures, rotor speeds, fuel ratios, bleed flows)

---

## ⚙️ Core Machine Learning Guidelines

1. **Zero Data Leakage**: Train/validation splits are strictly partitioned by `unit_number`, never by random record shuffling.
2. **Causal Time Constraints**: Rolling features are strictly backward-looking (`min_periods=1`, no future lookahead).
3. **Isolated Preprocessing**: Scalers and transformations are fitted exclusively on training units, then applied to validation and test units.
4. **Test Set Isolation**: Test set data and ground truth RUL are kept strictly unseen until final evaluation.
5. **No Target Contamination**: `RUL` is strictly treated as the target variable and never as an input feature.
6. **Reproducibility**: All stochastic procedures use fixed seed `random_state=42`.

---

## 🚀 Setup and Installation

### 1. Clone & Set Up Environment
```bash
git clone <repository_url>
cd turbofan-rul-ml
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Data Ingestion
Run the data loader or open the initial notebook:
```bash
python -c "from src.data_loader import load_cmapss_data; train, test, rul = load_cmapss_data(); print('Data loaded successfully!')"
```
