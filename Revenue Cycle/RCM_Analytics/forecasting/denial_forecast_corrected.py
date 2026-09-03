"""
RCM Denial Rate Forecasting — corrected pipeline
==================================================

Fixes applied vs. the original notebook:
  1. LEAKAGE REMOVED: denied_claims / total_claims (which directly compose
     denial_rate) are no longer used as model features. Only lag/rolling
     values of the target itself, plus calendar features, are used.
  2. RECURSIVE MULTI-STEP FORECASTING: the XGBoost future forecast no
     longer flatlines. Each future month is predicted one step at a time,
     fed back in as a lag feature for the next step.
  3. PARTIAL MONTH EXCLUDED: the most recent month is dropped if its claim
     volume is far below the historical average (a sign of a mid-month
     data cutoff).
  4. REAL WALK-FORWARD VALIDATION: model comparison now uses an expanding-
     window walk-forward CV loop (multiple folds), not a single 80/20
     holdout split.
  5. Prophet's yearly seasonality is disabled when there isn't enough
     history to support it (Prophet needs ~2 years for a stable yearly
     seasonality component).

Run this as a Databricks notebook/job (it uses `spark`, which Databricks
provides automatically) or adapt the `load_claims_data` function if running
outside Databricks.
"""

import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLAIMS_PATH = "/Workspace/Users/goladosurahman@gmail.com/Portfolio/Revenue Cycle/dataset/fact_claims.csv"
OUTPUT_TABLE = "workspace.default.denial_rate_forecast"

LAGS = [1, 2, 3, 6, 12]
FORECAST_HORIZON = 6          # months to forecast forward
MIN_TRAIN_MONTHS = 12         # minimum training window for walk-forward CV
PARTIAL_MONTH_RATIO = 0.5     # drop trailing month if volume < 50% of median volume


# See the full implementation in the corrected notebook: 01_denial_forecast_eda
# Key functions to implement:
# - load_claims_data, build_monthly_series, drop_partial_trailing_month
# - create_features (leakage-free)
# - run_walk_forward_cv (all 5 models)
# - recursive_xgboost_forecast (fixes flat forecast)
# - save_forecast_to_delta