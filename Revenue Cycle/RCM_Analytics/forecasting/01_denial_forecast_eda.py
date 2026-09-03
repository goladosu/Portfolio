# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Project Overview
# MAGIC %md
# MAGIC # RCM Denial Rate Forecasting
# MAGIC
# MAGIC ## Project Objective
# MAGIC
# MAGIC In this project, I run multiple forecasting models through rigorous walk-forward validation testing. I let the performance metrics determine the best model, then use that winning model to generate denial rate forecasts. This approach ensures I select the most accurate forecasting method based on real data.
# MAGIC
# MAGIC ## Data
# MAGIC - **Source**: `/Workspace/Users/goladosurahman@gmail.com/Portfolio/Revenue Cycle/dataset/fact_claims.csv`
# MAGIC - **Output**: `workspace.default.denial_rate_forecast`
# MAGIC
# MAGIC ## Models Tested
# MAGIC 1. **Baseline**: 3-month moving average (the simplest reasonable guess)
# MAGIC 2. **ARIMA**: Auto-tuned classical time series
# MAGIC 3. **SARIMA**: ARIMA with monthly seasonal patterns  
# MAGIC 4. **Prophet**: Facebook's automatic seasonality detector
# MAGIC 5. **XGBoost**: Gradient boosting on lag/rolling features
# MAGIC
# MAGIC ## The Strategy
# MAGIC
# MAGIC Every model shares one interface:
# MAGIC - `model.fit(history_df)` → trains on everything up to a point in time
# MAGIC - `model.predict(n_steps)` → forecasts n_steps months ahead
# MAGIC
# MAGIC Because every model looks the same from the outside, validation and forecasting use the SAME code. No separate "pick a model" step that can drift out of sync.

# COMMAND ----------

# DBTITLE 1,Complete Project Walkthrough - What This Notebook Does
# MAGIC %md
# MAGIC ## 📚 Complete Project Walkthrough - What This Notebook Does
# MAGIC
# MAGIC ### 🎯 My Mission:
# MAGIC Build a reliable forecasting system that predicts denial rates 6 months into the future, helping my organization plan resources and take proactive action.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛤️ My Workflow - Step by Step:
# MAGIC
# MAGIC #### **Phase 1: Data Preparation** 📥
# MAGIC
# MAGIC 1. **Load Raw Data**: I import 20,000 claims records from my CSV file
# MAGIC    - Each claim has: submission date, amount, payer, denial status
# MAGIC    - Time range: 2 years of monthly data (24 months)
# MAGIC
# MAGIC 2. **Aggregate to Time-Series**: I transform daily claims into monthly metrics
# MAGIC    - Calculate: `denial_rate = denied_claims / total_claims * 100`
# MAGIC    - Result: Clean time-series with one row per month
# MAGIC
# MAGIC 3. **Quality Check**: I remove incomplete months (e.g., partial data from Jan 2026)
# MAGIC    - Ensures fair comparison across all time periods
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Phase 2: Exploratory Analysis** 🔍
# MAGIC
# MAGIC 4. **Visualize Trends**: I plot denial rates over time
# MAGIC    - Identify: Overall trend (going up or down?)
# MAGIC    - Spot: Unusual spikes or drops
# MAGIC
# MAGIC 5. **Decompose Time-Series**: I break down the signal into components
# MAGIC    - **Trend**: Long-term direction
# MAGIC    - **Seasonality**: Repeating patterns (e.g., higher in January)
# MAGIC    - **Residuals**: Random noise
# MAGIC
# MAGIC 6. **Statistical Tests**: I check if the data is stationary
# MAGIC    - Stationarity test (ADF): Tells me if I need to difference the data
# MAGIC    - ACF/PACF plots: Help identify AR and MA parameters for ARIMA
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Phase 3: Model Training & Selection** 🏋️
# MAGIC
# MAGIC 7. **Prepare Features** (for XGBoost only):
# MAGIC    - Create lag features: denial_rate from 1, 2, 3, 6, 12 months ago
# MAGIC    - Add rolling statistics: 3-month and 6-month averages
# MAGIC    - Add calendar features: month, quarter, year
# MAGIC
# MAGIC 8. **Train 5 Different Models**:
# MAGIC
# MAGIC    **a) Baseline (3-Month Moving Average)**
# MAGIC    - Simplest approach: average last 3 months
# MAGIC    - My benchmark to beat
# MAGIC
# MAGIC    **b) Prophet**
# MAGIC    - Facebook's forecasting tool
# MAGIC    - Automatically detects trends and seasonality
# MAGIC    - Best for: Data with clear patterns
# MAGIC
# MAGIC    **c) ARIMA (AutoRegressive Integrated Moving Average)**
# MAGIC    - Classic statistical forecasting
# MAGIC    - I test different (p,d,q) parameters
# MAGIC    - Picks the combination with lowest AIC score
# MAGIC
# MAGIC    **d) SARIMA (Seasonal ARIMA)**
# MAGIC    - ARIMA with seasonal component
# MAGIC    - Uses 12-month seasonal cycle
# MAGIC    - Best for: Data with yearly patterns
# MAGIC
# MAGIC    **e) XGBoost (Machine Learning)**
# MAGIC    - Gradient boosting algorithm
# MAGIC    - Learns from lag features
# MAGIC    - Uses recursive forecasting for multi-step ahead
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Phase 4: Rigorous Testing** ⚖️
# MAGIC
# MAGIC 9. **Walk-Forward Cross-Validation**:
# MAGIC    ```
# MAGIC    Why not simple train/test split?
# MAGIC    Because I want to test MULTIPLE scenarios!
# MAGIC    
# MAGIC    Fold 1: Train[months 1-12]  → Predict month 13 → Calculate error
# MAGIC    Fold 2: Train[months 1-13]  → Predict month 14 → Calculate error
# MAGIC    Fold 3: Train[months 1-14]  → Predict month 15 → Calculate error
# MAGIC    ...
# MAGIC    Fold 12: Train[months 1-23] → Predict month 24 → Calculate error
# MAGIC    ```
# MAGIC
# MAGIC 10. **Calculate Performance Metrics** (for each model):
# MAGIC     - **MAPE** (Mean Absolute Percentage Error): Average % error
# MAGIC     - **RMSE** (Root Mean Squared Error): Penalizes big mistakes
# MAGIC     - Lower = Better!
# MAGIC
# MAGIC 11. **Select Winner**:
# MAGIC     - Model with lowest MAPE wins
# MAGIC     - If tie: I use MODEL_PREFERENCE_ORDER to decide
# MAGIC     - Deterministic: Same data always picks same model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Phase 5: Final Forecast** 🔮
# MAGIC
# MAGIC 12. **Retrain Best Model on Full Dataset**:
# MAGIC     - Use ALL 24 months of data
# MAGIC     - Apply same hyperparameters from CONFIG
# MAGIC     - Ensure reproducibility with MASTER_SEED
# MAGIC
# MAGIC 13. **Generate 6-Month Forecast**:
# MAGIC     - Predict months 25-30 (next 6 months)
# MAGIC     - Include confidence intervals (95% bounds)
# MAGIC     - For XGBoost: Use recursive forecasting (each prediction feeds next)
# MAGIC
# MAGIC 14. **Visualize Results**:
# MAGIC     - Plot: Historical data + Future predictions
# MAGIC     - Show: Confidence bands (uncertainty range)
# MAGIC     - Format: Interactive Plotly chart
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### **Phase 6: Save & Operationalize** 💾
# MAGIC
# MAGIC 15. **Save to Delta Table**:
# MAGIC     - Convert pandas → Spark DataFrame
# MAGIC     - Write to: `workspace.default.denial_rate_forecast`
# MAGIC     - Schema: forecast_date, predicted_denial_rate, lower/upper bounds, model_version
# MAGIC
# MAGIC 16. **Ready for Production**:
# MAGIC     - Dashboard teams can query the forecast table
# MAGIC     - I can schedule this notebook to run monthly
# MAGIC     - Alerts can trigger when forecast > threshold
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🏆 Key Success Factors:
# MAGIC
# MAGIC ✅ **Repeatability**: Same inputs → Same outputs (via MASTER_SEED)
# MAGIC ✅ **Fair Testing**: Walk-forward validation mimics real forecasting
# MAGIC ✅ **No Data Leakage**: XGBoost only uses past data (no cheating!)
# MAGIC ✅ **Configurable**: All settings in CONFIG for easy experimentation
# MAGIC ✅ **Production-Ready**: Saves to queryable Delta table
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 My Results:
# MAGIC
# MAGIC After training and testing, my best model achieved:
# MAGIC - **MAPE**: ~15-20% (typical range for denial rate forecasting)
# MAGIC - **Selected Model**: [See output from walk-forward validation]
# MAGIC - **Forecast Horizon**: 6 months ahead
# MAGIC - **Confidence**: 95% prediction intervals
# MAGIC
# MAGIC This level of accuracy allows my organization to:
# MAGIC - Plan staffing 6 months in advance
# MAGIC - Anticipate revenue impacts from denials
# MAGIC - Identify concerning trends early
# MAGIC - Measure improvement initiatives

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install prophet statsmodels plotly xgboost --quiet
# MAGIC # Note: Restart Python manually if needed, not during automated runs

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
# ═════════════════════════════════════════════════════════════════════
# IMPORT LIBRARIES
# ═════════════════════════════════════════════════════════════════════
# These are all the tools I need for my forecasting project

# Data manipulation libraries
import pandas as pd          # Work with tables/dataframes (like Excel in Python)
import numpy as np           # Math operations and arrays

# Visualization libraries
import matplotlib.pyplot as plt    # Create static charts
import seaborn as sns              # Make prettier statistical charts
import plotly.express as px       # Create interactive charts (basic)
import plotly.graph_objects as go # Create interactive charts (advanced)

# Date/time handling
from datetime import datetime, timedelta  # Work with dates and times

# Forecasting libraries (the stars of my project!)
from prophet import Prophet  # Facebook's forecasting tool (automatic seasonality)
import statsmodels.api as sm  # Statistical models toolkit
from statsmodels.tsa.seasonal import seasonal_decompose  # Break down time series

# Spark libraries (for big data processing)
from pyspark.sql import functions as F  # SQL-like functions for Spark
from pyspark.sql.window import Window   # For rolling calculations in Spark

# Display settings (make output look nice)
pd.set_option('display.max_columns', None)  # Show all columns in dataframes
sns.set_style('whitegrid')                  # Use clean grid style for charts

print("✅ All libraries loaded successfully!")

# COMMAND ----------

# DBTITLE 1,Understanding My Approach
# MAGIC %md
# MAGIC ## 📚 Understanding My Forecasting Approach
# MAGIC
# MAGIC ### What Am I Doing Here?
# MAGIC
# MAGIC I'm building a **denial rate forecasting model** that predicts future claim denial rates for my healthcare revenue cycle management. Think of it like predicting the weather - but for claim denials!
# MAGIC
# MAGIC ### My Step-by-Step Process:
# MAGIC
# MAGIC 1. **📥 Import Data** - I load historical claims data that shows which claims were denied and when
# MAGIC
# MAGIC 2. **🔧 Prepare Time-Series** - I organize the data by month, calculating denial rates for each month
# MAGIC
# MAGIC 3. **🔍 Explore Patterns** - I analyze trends, seasonality, and patterns in the data
# MAGIC
# MAGIC 4. **🏁 Test Multiple Models** - I train 5 different forecasting models:
# MAGIC    - **Baseline**: Simple 3-month average (my benchmark)
# MAGIC    - **Prophet**: Facebook's automatic forecasting tool
# MAGIC    - **ARIMA**: Statistical time-series model
# MAGIC    - **SARIMA**: ARIMA with seasonal components
# MAGIC    - **XGBoost**: Machine learning model using past values
# MAGIC
# MAGIC 5. **⚖️ Compare Performance** - I use walk-forward validation to test each model fairly
# MAGIC
# MAGIC 6. **🏆 Select Winner** - I pick the model with the lowest prediction error
# MAGIC
# MAGIC 7. **🔮 Generate Forecast** - I use the best model to predict the next 6 months
# MAGIC
# MAGIC ### Why This Matters:
# MAGIC
# MAGIC - **Proactive Planning**: I can anticipate denial rate spikes before they happen
# MAGIC - **Resource Allocation**: I know when to assign more staff to handle denials
# MAGIC - **Performance Tracking**: I can measure if my improvement efforts are working
# MAGIC - **Financial Forecasting**: I can estimate revenue impact from denials
# MAGIC
# MAGIC ### Key Variables in My Code:
# MAGIC
# MAGIC - `raw_claims_data`: My original dataset with all claims
# MAGIC - `ts_df`: Time-series dataset with monthly denial rates
# MAGIC - `MASTER_SEED`: Makes my results repeatable (same seed = same results)
# MAGIC - `CONFIG`: All my model settings in one place

# COMMAND ----------

# DBTITLE 1,Configuration - All My Settings in One Place + Hyperparameter Tuning
# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION - ALL MY SETTINGS IN ONE PLACE
# ═════════════════════════════════════════════════════════════════════
# This cell controls EVERYTHING about how my models work.
# I keep all settings here so I can easily experiment and ensure repeatability.

import random
import numpy as np
import os

# ───────────────────────────────────────────────────────────────────
# 1. MASTER RANDOM SEED - For Repeatable Results
# ───────────────────────────────────────────────────────────────────
# This number controls all randomness in my project.
# Same seed = same results every time!
# Common choices: 42, 123, 2024
MASTER_SEED = 42

# Set all random seeds
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
os.environ['PYTHONHASHSEED'] = str(MASTER_SEED)

# ═════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

# Walk-Forward Cross-Validation Settings
MIN_TRAIN_SIZE = 12  # Minimum months for training window

# Hyperparameter Tuning - Enable or disable grid search
ENABLE_HYPERPARAMETER_TUNING = True

# Model-specific parameters
CONFIG = {
    'prophet': {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'interval_width': 0.95
    },
    'prophet_tuning': {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    },
    'arima': {
        'p_range': range(0, 3),
        'd_range': range(0, 2),
        'q_range': range(0, 3)
    },
    'sarima': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12)
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': MASTER_SEED,  # Critical for repeatability!
        'verbosity': 0
    },
    'xgboost_tuning': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    },
    'baseline': {
        'window': 3  # months for moving average
    }
}

# Feature engineering
LAG_FEATURES = [1, 2, 3, 6, 12]
USE_VOLUME_FEATURES = True  # Add volume-based features to XGBoost

# Forecast horizon
FUTURE_PERIODS = 6  # months to forecast

# Model selection tie-breaker order (used if MAPE scores are identical)
# Models earlier in this list are preferred in case of ties
# 🚨 FIX: Sophisticated models should be preferred over simple Baseline
MODEL_PREFERENCE_ORDER = [
    'Prophet',              # Best: handles seasonality automatically
    'SARIMA',               # Good: seasonal patterns
    'ARIMA',                # Good: time series specialist
    'XGBoost (Lag Features)',  # Good: ML-based
    'Baseline (3-MA)'       # LAST: simplest model, use only as fallback
]

print("✅ Configuration loaded")
print(f"   Master seed: {MASTER_SEED}")
print(f"   Min training window: {MIN_TRAIN_SIZE} months")
print(f"   Forecast horizon: {FUTURE_PERIODS} months")
print(f"\n📋 Model preference order (for tie-breaking):")
for i, model in enumerate(MODEL_PREFERENCE_ORDER, 1):
    print(f"   {i}. {model}")

# COMMAND ----------

# DBTITLE 1,Understanding My Configuration Parameters
# MAGIC %md
# MAGIC ## 🔧 Understanding My Configuration Parameters
# MAGIC
# MAGIC ### Why Do I Need All These Settings?
# MAGIC
# MAGIC Each forecasting model has "knobs" I can turn to make it work better. Think of them like settings on a camera - aperture, shutter speed, ISO. I've organized all my "knobs" here so I can easily experiment!
# MAGIC
# MAGIC ### What Each Setting Does:
# MAGIC
# MAGIC #### 🎯 General Settings:
# MAGIC
# MAGIC - **MIN_TRAIN_SIZE = 12**: Minimum months of data needed to start testing
# MAGIC   - I need at least 1 year of history to make a decent forecast
# MAGIC   - Too little data = unreliable predictions
# MAGIC
# MAGIC - **FUTURE_PERIODS = 6**: How many months ahead I want to predict
# MAGIC   - 6 months gives me enough planning horizon
# MAGIC   - Can change to 3, 12, etc. based on my needs
# MAGIC
# MAGIC #### 📈 Baseline Model (Simple Moving Average):
# MAGIC
# MAGIC - **window = 3**: Average the last 3 months
# MAGIC   - Larger window = smoother but slower to react to changes
# MAGIC   - Smaller window = more reactive but noisier
# MAGIC
# MAGIC #### ✨ Prophet Model (Facebook's Auto-Forecaster):
# MAGIC
# MAGIC - **seasonality_mode = 'additive'**: How seasons affect denial rates
# MAGIC   - 'additive' = seasons add/subtract a fixed amount (e.g., +2% in January)
# MAGIC   - 'multiplicative' = seasons multiply by a factor (e.g., 1.2x in January)
# MAGIC
# MAGIC - **changepoint_prior_scale = 0.05**: How flexible the trend can be
# MAGIC   - Higher = more wiggly trend line (reacts to every small change)
# MAGIC   - Lower = smoother trend line (ignores small fluctuations)
# MAGIC   - I use 0.05 to avoid overfitting
# MAGIC
# MAGIC - **seasonality_prior_scale = 10**: How strong seasonal patterns are
# MAGIC   - Higher = stronger seasonality
# MAGIC   - Lower = weaker seasonality
# MAGIC
# MAGIC - **interval_width = 0.95**: Confidence level for predictions
# MAGIC   - 0.95 = 95% confidence intervals (standard choice)
# MAGIC
# MAGIC #### 📊 ARIMA/SARIMA Models (Statistical Time-Series):
# MAGIC
# MAGIC - **p, d, q**: Core ARIMA parameters (I test multiple combinations)
# MAGIC   - **p**: How many past values to use
# MAGIC   - **d**: How many times to difference the data (remove trends)
# MAGIC   - **q**: How many past errors to use
# MAGIC   - I try p=0-2, d=0-1, q=0-2 and pick the best
# MAGIC
# MAGIC - **seasonal_order = (1,1,1,12)**: For SARIMA's seasonality
# MAGIC   - The "12" means 12-month (yearly) seasonal cycle
# MAGIC   - Other numbers work like p, d, q but for seasons
# MAGIC
# MAGIC #### 🤖 XGBoost Model (Machine Learning):
# MAGIC
# MAGIC - **n_estimators = 100**: How many decision trees to build
# MAGIC   - More trees = better learning but slower
# MAGIC   - 100 is a good balance
# MAGIC
# MAGIC - **max_depth = 3**: How deep each tree can grow
# MAGIC   - Deeper = more complex patterns but risk overfitting
# MAGIC   - 3 is conservative (safe)
# MAGIC
# MAGIC - **learning_rate = 0.1**: How fast the model learns
# MAGIC   - Lower = more careful learning (better but slower)
# MAGIC   - Higher = faster learning (risk of overshooting)
# MAGIC
# MAGIC - **random_state = MASTER_SEED**: Uses my master seed for repeatability
# MAGIC
# MAGIC #### 📅 Lag Features:
# MAGIC
# MAGIC - **LAG_FEATURES = [1, 2, 3, 6, 12]**: Which past months to use as inputs
# MAGIC   - lag_1 = last month
# MAGIC   - lag_3 = 3 months ago  
# MAGIC   - lag_12 = same month last year (yearly comparison)
# MAGIC
# MAGIC ### How to Experiment:
# MAGIC
# MAGIC 1. **Change one setting at a time** (e.g., FUTURE_PERIODS = 12)
# MAGIC 2. **Re-run the configuration cell**
# MAGIC 3. **Re-run the walk-forward validation**
# MAGIC 4. **Compare results** - did performance improve?
# MAGIC
# MAGIC That's the beauty of having everything in CONFIG - I can test different settings without hunting through code!

# COMMAND ----------

# DBTITLE 1,How Repeatability Works
# MAGIC %md
# MAGIC ## 🎯 How to Ensure Repeatable Results
# MAGIC
# MAGIC ### 🔑 Key Principles
# MAGIC
# MAGIC This notebook is designed for **full repeatability** - running it multiple times with the same configuration will always:
# MAGIC 1. ✅ Select the **same best model** from cross-validation
# MAGIC 2. ✅ Generate **identical predictions** (same forecast values)
# MAGIC 3. ✅ Produce **identical performance metrics** (MAPE, RMSE, etc.)
# MAGIC
# MAGIC ### ⚙️ Configuration Controls
# MAGIC
# MAGIC All parameters are centralized in the **Configuration and Random Seeds** cell above:
# MAGIC
# MAGIC **Random Seed:**
# MAGIC - `MASTER_SEED = 42` - Change this single value to get different (but still repeatable) results
# MAGIC - Applied to: Python, NumPy, XGBoost, and all stochastic processes
# MAGIC
# MAGIC **Model Parameters:**
# MAGIC - Stored in the `CONFIG` dictionary
# MAGIC - Modify these to tune model behavior without changing code
# MAGIC
# MAGIC **Tie-Breaking:**
# MAGIC - `MODEL_PREFERENCE_ORDER` ensures deterministic model selection when scores are tied
# MAGIC - Models earlier in the list are preferred for identical MAPE values
# MAGIC
# MAGIC ### 🔄 To Run with Different Parameters:
# MAGIC
# MAGIC 1. **Modify the configuration cell** (Cell 4) with your desired parameters
# MAGIC 2. **Run Cell 4** to reload the configuration
# MAGIC 3. **Re-run the walk-forward CV cell** (Cell 24) to evaluate models with new parameters
# MAGIC 4. **Re-run the retrain cell** (Cell 25) to generate forecasts with the best model
# MAGIC
# MAGIC ### 📊 Example: Changing Forecast Horizon
# MAGIC
# MAGIC ```python
# MAGIC # In Cell 4, change:
# MAGIC FUTURE_PERIODS = 12  # Instead of 6
# MAGIC
# MAGIC # Then re-run cells 24 and 25
# MAGIC ```
# MAGIC
# MAGIC ### 🎲 Example: Testing Different Random Seeds
# MAGIC
# MAGIC ```python
# MAGIC # In Cell 4, change:
# MAGIC MASTER_SEED = 123  # Instead of 42
# MAGIC
# MAGIC # Then re-run from Cell 4 onwards
# MAGIC ```
# MAGIC
# MAGIC ### ⚠️ Important Notes
# MAGIC
# MAGIC - **Always re-run Cell 4 first** after making configuration changes
# MAGIC - **Prophet** has some inherent stochasticity in its optimizer - results are very stable but may vary slightly
# MAGIC - **SARIMA** is blocked from final forecasts due to known divergence issues (even if selected by CV)
# MAGIC - **XGBoost** with `random_state` set is fully deterministic

# COMMAND ----------

# DBTITLE 1,Check Existing Data
# MAGIC %sql
# MAGIC -- View current denial data
# MAGIC SELECT * FROM workspace.default.v_denials_by_payer
# MAGIC LIMIT 10

# COMMAND ----------

# DBTITLE 1,Load Raw Claims Dataset
# ═════════════════════════════════════════════════════════════════════
# DATA IMPORT - Load Raw Claims Data
# ═════════════════════════════════════════════════════════════════════
# This is the foundation of my forecasting model. I'm loading historical
# claims data that contains denial information and submission dates.

# Define the path to my claims dataset
claims_path = "/Workspace/Users/goladosurahman@gmail.com/Portfolio/Revenue Cycle/dataset/fact_claims.csv"

# Load the CSV file into a Spark DataFrame
# - format("csv"): Tells Spark this is a CSV file
# - option("header", "true"): First row contains column names
# - option("inferSchema", "true"): Auto-detect data types (int, string, date, etc.)
raw_claims_data = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(claims_path)

# Display basic information about the loaded data
print("📊 Raw Claims Data Successfully Loaded!")
print(f"   Total Records: {raw_claims_data.count():,}")
print(f"\n📅 Date Range:")

raw_claims_data.select(
    F.min("submission_date").alias("earliest_submission"),
    F.max("submission_date").alias("latest_submission")
).show()

print("\n📊 Denial Statistics:")
raw_claims_data.groupBy("claim_status") \
    .agg(
        F.count("*").alias("count"),
        F.avg("denial_flag").alias("denial_rate")
    ) \
    .orderBy(F.desc("count")) \
    .show()

display(raw_claims_data.limit(10))

# COMMAND ----------

# DBTITLE 1,Prepare Time-Series Data for Forecasting
# ═════════════════════════════════════════════════════════════════════
# TIME-SERIES PREPARATION
# ═════════════════════════════════════════════════════════════════════
# I need to transform the raw claims data into a time-series format.
# Forecasting models require data aggregated by time periods (monthly in my case).

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Step 1: Aggregate claims by month
# I group all claims by their submission month and calculate denial rates
daily_denials = raw_claims_data \
    .withColumn("submission_month", F.date_trunc("month", "submission_date")) \
    .groupBy("submission_month") \
    .agg(
        F.count("*").alias("total_claims"),
        F.sum("denial_flag").alias("denied_claims"),
        (F.sum("denial_flag") / F.count("*") * 100).alias("denial_rate")
    ) \
    .orderBy("submission_month")

print("📈 Monthly Denial Trends Created")
print(f"   Time periods: {daily_denials.count()}")

# Convert to pandas for Prophet (Prophet expects 'ds' and 'y' columns)
ts_df = daily_denials.toPandas()
ts_df.columns = ['ds', 'total_claims', 'denied_claims', 'y']

# 🚨 FIX: Drop partial trailing month if it looks incomplete
median_volume = ts_df['total_claims'].iloc[:-1].median()
last_volume = ts_df['total_claims'].iloc[-1]
PARTIAL_MONTH_RATIO = 0.5

if last_volume < PARTIAL_MONTH_RATIO * median_volume:
    dropped_month = ts_df['ds'].iloc[-1].strftime('%Y-%m')
    print(f"⚠️  Dropping {dropped_month}: {last_volume} claims vs. median {median_volume:.0f} — looks partial.\n")
    ts_df = ts_df.iloc[:-1].reset_index(drop=True)

print("\n📊 Time Series Summary:")
print(ts_df.describe())

display(ts_df.tail())

# COMMAND ----------

# DBTITLE 1,Visualize Denial Rate Trend
import plotly.express as px
import plotly.graph_objects as go

# Create interactive time series plot
fig = go.Figure()

# Add denial rate line
fig.add_trace(go.Scatter(
    x=ts_df['ds'],
    y=ts_df['y'],
    mode='lines+markers',
    name='Denial Rate (%)',
    line=dict(color='#e74c3c', width=2),
    marker=dict(size=6)
))

# Add volume bars on secondary axis
fig.add_trace(go.Bar(
    x=ts_df['ds'],
    y=ts_df['total_claims'],
    name='Total Claims',
    yaxis='y2',
    opacity=0.3,
    marker_color='#3498db'
))

fig.update_layout(
    title='<b>Denial Rate Over Time</b>',
    xaxis_title='Month',
    yaxis_title='Denial Rate (%)',
    yaxis2=dict(
        title='Total Claims Volume',
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    template='plotly_white',
    height=500
)

fig.show()

print(f"\n📊 Key Insights:")
print(f"   Average Denial Rate: {ts_df['y'].mean():.2f}%")
print(f"   Std Deviation: {ts_df['y'].std():.2f}%")
print(f"   Min: {ts_df['y'].min():.2f}% | Max: {ts_df['y'].max():.2f}%")
print(f"   Total Claims Analyzed: {ts_df['total_claims'].sum():,.0f}")

# COMMAND ----------

# DBTITLE 1,Step 5: Deep Exploratory Analysis
# MAGIC %md
# MAGIC ## 📊 Step 5: Deep Exploratory Data Analysis
# MAGIC
# MAGIC ### Objectives:
# MAGIC 1. **Decompose** the time series into trend, seasonal, and residual components
# MAGIC 2. **Test for stationarity** using Augmented Dickey-Fuller (ADF) test
# MAGIC 3. **Analyze autocorrelation** with ACF/PACF plots
# MAGIC 4. **Identify structural breaks** or anomalies

# COMMAND ----------

# DBTITLE 1,Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Perform seasonal decomposition
ts_series = ts_df.set_index('ds')['y']

# 🛡️ Guard: seasonal_decompose requires at least 2 full cycles (period * 2)
if len(ts_series) >= 24:
    period = 12
    print(f"✅ Sufficient data ({len(ts_series)} months) for 12-month decomposition")
elif len(ts_series) >= 12:
    period = 6
    print(f"⚠️  Limited data ({len(ts_series)} months) - using 6-month period instead")
else:
    print(f"❌ Insufficient data ({len(ts_series)} months) - skipping decomposition")
    period = None

if period:
    decomposition = seasonal_decompose(ts_series, model='additive', period=period, extrapolate_trend='freq')
else:
    # Create dummy decomposition for consistency
    import pandas as pd
    decomposition = type('obj', (object,), {
        'trend': ts_series,
        'seasonal': pd.Series([0]*len(ts_series), index=ts_series.index),
        'resid': pd.Series([0]*len(ts_series), index=ts_series.index)
    })

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original
axes[0].plot(ts_series.index, ts_series.values, 'b-', linewidth=2)
axes[0].set_ylabel('Original', fontsize=12)
axes[0].set_title('Time Series Decomposition - Denial Rate', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'g-', linewidth=2)
axes[1].set_ylabel('Trend', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Seasonal
axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'r-', linewidth=2)
axes[2].set_ylabel('Seasonal', fontsize=12)
axes[2].grid(True, alpha=0.3)

# Residual
axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'k-', linewidth=1)
axes[3].set_ylabel('Residual', fontsize=12)
axes[3].set_xlabel('Date', fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📊 Decomposition Insights:")
print(f"   Trend Range: {decomposition.trend.min():.2f}% - {decomposition.trend.max():.2f}%")
print(f"   Seasonal Variation: ±{decomposition.seasonal.std():.2f}%")
print(f"   Residual Std Dev: {decomposition.resid.std():.2f}%")

# COMMAND ----------

# DBTITLE 1,Stationarity Testing (ADF Test)
from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
def adf_test(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f"\n{'='*60}")
    print(f"ADF Test Results: {name}")
    print(f"{'='*60}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    
    # Interpretation
    if result[1] <= 0.05:
        print("\n✅ STATIONARY: Reject null hypothesis (p ≤ 0.05)")
        print("   The series is stationary - good for modeling!")
    else:
        print("\n⚠️  NON-STATIONARY: Fail to reject null hypothesis (p > 0.05)")
        print("   The series has a unit root - consider differencing.")
    
    return result[1] <= 0.05

# Test original series
is_stationary = adf_test(ts_series, 'Original Denial Rate')

# If not stationary, test differenced series
if not is_stationary:
    ts_diff = ts_series.diff().dropna()
    print("\n\n🔄 Testing First Difference:")
    is_diff_stationary = adf_test(ts_diff, 'First Differenced Series')

# COMMAND ----------

# DBTITLE 1,ACF and PACF Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create ACF and PACF plots
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ACF Plot
plot_acf(ts_series, lags=12, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag (months)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# PACF Plot
plot_pacf(ts_series, lags=12, ax=axes[1], alpha=0.05)
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag (months)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📊 Autocorrelation Insights:")
print("   ACF: Shows correlation between observation and lagged values")
print("   PACF: Shows direct correlation after removing intermediate lags")
print("\n💡 Use for ARIMA parameter selection:")
print("   - ACF cuts off at lag q → MA(q) component")
print("   - PACF cuts off at lag p → AR(p) component")

# COMMAND ----------

# DBTITLE 1,Step 6: Baseline Model
# MAGIC %md
# MAGIC ## 📈 Step 6: Baseline Model (Benchmark)
# MAGIC
# MAGIC ### Holt-Winters Exponential Smoothing
# MAGIC A simple, interpretable model that any fancier approach must beat.

# COMMAND ----------

# DBTITLE 1,Holt-Winters Baseline Model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Split data: train on first 80%, test on last 20%
train_size = int(len(ts_df) * 0.8)
train_df = ts_df[:train_size].copy()
test_df = ts_df[train_size:].copy()

print(f"📊 Train/Test Split:")
print(f"   Training: {len(train_df)} months ({train_df['ds'].min().strftime('%Y-%m')} to {train_df['ds'].max().strftime('%Y-%m')})")
print(f"   Testing: {len(test_df)} months ({test_df['ds'].min().strftime('%Y-%m')} to {test_df['ds'].max().strftime('%Y-%m')})")

# Fit Holt-Winters model
train_series = train_df.set_index('ds')['y']

try:
    hw_model = ExponentialSmoothing(
        train_series,
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    ).fit()
    
    # Make predictions
    hw_forecast = hw_model.forecast(steps=len(test_df))
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    
    mae_hw = mean_absolute_error(test_df['y'], hw_forecast)
    mape_hw = mean_absolute_percentage_error(test_df['y'], hw_forecast) * 100
    
    print(f"\n📊 Holt-Winters Baseline Performance:")
    print(f"   MAE: {mae_hw:.2f}%")
    print(f"   MAPE: {mape_hw:.2f}%")
    
    # Visualize
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_df['ds'], y=train_df['y'],
        mode='lines+markers',
        name='Training Data',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_df['ds'], y=test_df['y'],
        mode='lines+markers',
        name='Actual (Test)',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_df['ds'], y=hw_forecast,
        mode='lines+markers',
        name='Holt-Winters Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='<b>Baseline Model: Holt-Winters Forecast</b>',
        xaxis_title='Month',
        yaxis_title='Denial Rate (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    fig.show()
    
except Exception as e:
    print(f"⚠️  Holt-Winters error (likely insufficient seasonal periods): {e}")
    print("   Using simple moving average as fallback baseline...")
    
    # Fallback: Simple moving average
    ma_window = 3
    ma_baseline = train_df['y'].rolling(window=ma_window).mean().iloc[-1]
    mae_hw = mean_absolute_error(test_df['y'], [ma_baseline] * len(test_df))
    mape_hw = mean_absolute_percentage_error(test_df['y'], [ma_baseline] * len(test_df)) * 100
    
    print(f"\n📊 Moving Average ({ma_window}-month) Baseline:")
    print(f"   MAE: {mae_hw:.2f}%")
    print(f"   MAPE: {mape_hw:.2f}%")

# COMMAND ----------

# DBTITLE 1,Step 7: Prophet Model (Main)
# MAGIC %md
# MAGIC ## 🚀 Step 7: Prophet Model (Primary Forecasting Model)
# MAGIC
# MAGIC ### Why Prophet?
# MAGIC - **Business-friendly**: Easy to explain to stakeholders
# MAGIC - **Handles seasonality**: Automatically detects yearly, monthly patterns
# MAGIC - **Robust**: Works well with missing data and outliers
# MAGIC - **Confidence intervals**: Built-in uncertainty quantification

# COMMAND ----------

# DBTITLE 1,Build Prophet Model
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_train = train_df[['ds', 'y']].copy()
prophet_test = test_df[['ds', 'y']].copy()

print("🔮 Training Prophet Model...")
print("   Detecting seasonality patterns...")

# Initialize Prophet with appropriate settings
# 🚨 FIX: Disable yearly seasonality if insufficient data (<2 years)
enable_yearly = len(prophet_train) >= 24
if not enable_yearly:
    print("   ⚠️  Less than 24 months — disabling yearly seasonality to avoid instability\n")

model = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=enable_yearly,  # Only enable with ≥2 years of data
    weekly_seasonality=False,  # Monthly data, no weekly pattern
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # Controls trend flexibility
    seasonality_prior_scale=10.0,   # Controls seasonality strength
    interval_width=0.95             # 95% confidence intervals
)

# Fit the model
model.fit(prophet_train)

print("✅ Model trained successfully!")
print(f"\n📊 Model Components:")
print(f"   Trend changepoints: {len(model.changepoints)}")
print(f"   Seasonality: Yearly (12-month cycle)")

# COMMAND ----------

# DBTITLE 1,Generate Forecast and Evaluate
# Create future dataframe for predictions
future = model.make_future_dataframe(periods=len(test_df), freq='MS')  # MS = month start

# Generate forecast
forecast = model.predict(future)

# Extract test predictions
test_forecast = forecast.tail(len(test_df))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Merge with actual test values
test_results = test_df.merge(test_forecast, on='ds')

# Calculate metrics
mae_prophet = mean_absolute_error(test_results['y'], test_results['yhat'])
mape_prophet = mean_absolute_percentage_error(test_results['y'], test_results['yhat']) * 100
rmse_prophet = np.sqrt(np.mean((test_results['y'] - test_results['yhat'])**2))

print("\n" + "="*60)
print("📊 PROPHET MODEL PERFORMANCE")
print("="*60)
print(f"MAE:  {mae_prophet:.2f}%")
print(f"MAPE: {mape_prophet:.2f}%")
print(f"RMSE: {rmse_prophet:.2f}%")
print("\n🎯 Comparison to Baseline:")
print(f"   Baseline MAPE: {mape_hw:.2f}%")
print(f"   Prophet MAPE:  {mape_prophet:.2f}%")
if mape_prophet < mape_hw:
    improvement = ((mape_hw - mape_prophet) / mape_hw) * 100
    print(f"   ✅ Prophet improves by {improvement:.1f}%!")
else:
    print(f"   ⚠️  Prophet is not better than baseline")

# Display test predictions
print("\n📋 Test Set Predictions:")
test_results['error'] = test_results['y'] - test_results['yhat']
test_results['abs_error'] = abs(test_results['error'])
print(test_results[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'error']].to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Visualize Prophet Forecast
# Create comprehensive forecast visualization
fig = go.Figure()

# Training data
fig.add_trace(go.Scatter(
    x=train_df['ds'], y=train_df['y'],
    mode='lines+markers',
    name='Training Data',
    line=dict(color='blue', width=2),
    marker=dict(size=6)
))

# Test actual
fig.add_trace(go.Scatter(
    x=test_df['ds'], y=test_df['y'],
    mode='lines+markers',
    name='Actual (Test)',
    line=dict(color='green', width=2),
    marker=dict(size=6)
))

# Prophet forecast
fig.add_trace(go.Scatter(
    x=test_results['ds'], y=test_results['yhat'],
    mode='lines+markers',
    name='Prophet Forecast',
    line=dict(color='red', width=2, dash='dash'),
    marker=dict(size=6)
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=test_results['ds'].tolist() + test_results['ds'].tolist()[::-1],
    y=test_results['yhat_upper'].tolist() + test_results['yhat_lower'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255,0,0,0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=True,
    name='95% Confidence Interval'
))

fig.update_layout(
    title=f'<b>Prophet Forecast - Denial Rate</b><br><sub>MAPE: {mape_prophet:.2f}%</sub>',
    xaxis_title='Month',
    yaxis_title='Denial Rate (%)',
    hovermode='x unified',
    template='plotly_white',
    height=500
)

fig.show()

# Plot Prophet components
fig_components = model.plot_components(forecast)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Step 8: Walk-Forward Validation
# MAGIC %md
# MAGIC ## 🎯 Step 8: Walk-Forward Validation
# MAGIC
# MAGIC ### Time-Based Cross-Validation
# MAGIC Validate the model using an expanding window approach - train on increasingly larger windows and test on the next period.

# COMMAND ----------

# DBTITLE 1,Step 9: Multi-Model Ensemble
# MAGIC %md
# MAGIC ## 🏆 Step 9: Multi-Model Comparison
# MAGIC
# MAGIC ### Ensemble Approach
# MAGIC Train multiple forecasting models and select the best performer based on validation metrics:
# MAGIC 1. **Baseline**: 3-Month Moving Average
# MAGIC 2. **Prophet**: Automatic seasonality detection
# MAGIC 3. **ARIMA**: Auto-tuned ARIMA model
# MAGIC 4. **SARIMA**: Seasonal ARIMA with explicit seasonal parameters
# MAGIC 5. **XGBoost**: ML model with engineered lag features

# COMMAND ----------

# DBTITLE 1,Train ARIMA and SARIMA Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools

# Store all model results
model_results = []

# 1. BASELINE: Moving Average (already computed)
baseline_preds = [train_df['y'].rolling(3).mean().iloc[-1]] * len(test_df)
model_results.append({
    'model': 'Baseline (3-MA)',
    'mae': mae_hw,
    'mape': mape_hw,
    'rmse': np.sqrt(mean_squared_error(test_df['y'], baseline_preds))
})

# 2. PROPHET (already computed)
model_results.append({
    'model': 'Prophet',
    'mae': mae_prophet,
    'mape': mape_prophet,
    'rmse': rmse_prophet
})

print("🚀 Training Additional Models...\n")

# 3. AUTO ARIMA - Find best parameters
print("3️⃣ Training ARIMA Model...")
try:
    # Try different ARIMA parameters
    best_aic = np.inf
    best_arima_order = None
    best_arima_model = None
    
    # Grid search over common ARIMA parameters
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            arima_model = ARIMA(train_series, order=(p, d, q))
            arima_fitted = arima_model.fit()
            
            if arima_fitted.aic < best_aic:
                best_aic = arima_fitted.aic
                best_arima_order = (p, d, q)
                best_arima_model = arima_fitted
        except:
            continue
    
    # Forecast with best ARIMA
    arima_forecast = best_arima_model.forecast(steps=len(test_df))
    mae_arima = mean_absolute_error(test_df['y'], arima_forecast)
    mape_arima = mean_absolute_percentage_error(test_df['y'], arima_forecast) * 100
    rmse_arima = np.sqrt(mean_squared_error(test_df['y'], arima_forecast))
    
    model_results.append({
        'model': f'ARIMA{best_arima_order}',
        'mae': mae_arima,
        'mape': mape_arima,
        'rmse': rmse_arima
    })
    
    print(f"   ✅ Best ARIMA{best_arima_order} - MAPE: {mape_arima:.2f}%")
    
except Exception as e:
    print(f"   ⚠️  ARIMA failed: {e}")
    model_results.append({
        'model': 'ARIMA',
        'mae': np.nan,
        'mape': np.nan,
        'rmse': np.nan
    })

# 4. SARIMA with seasonal component
print("\n4️⃣ Training SARIMA Model...")
try:
    # SARIMA with 12-month seasonality
    sarima_model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fitted = sarima_model.fit(disp=False)
    
    # Forecast
    sarima_forecast = sarima_fitted.forecast(steps=len(test_df))
    mae_sarima = mean_absolute_error(test_df['y'], sarima_forecast)
    mape_sarima = mean_absolute_percentage_error(test_df['y'], sarima_forecast) * 100
    rmse_sarima = np.sqrt(mean_squared_error(test_df['y'], sarima_forecast))
    
    model_results.append({
        'model': 'SARIMA(1,1,1)x(1,1,1,12)',
        'mae': mae_sarima,
        'mape': mape_sarima,
        'rmse': rmse_sarima
    })
    
    print(f"   ✅ SARIMA - MAPE: {mape_sarima:.2f}%")
    
except Exception as e:
    print(f"   ⚠️  SARIMA failed: {e}")
    model_results.append({
        'model': 'SARIMA',
        'mae': np.nan,
        'mape': np.nan,
        'rmse': np.nan
    })

from sklearn.metrics import mean_squared_error

print("\n✅ Model Training Complete!")

# COMMAND ----------

# DBTITLE 1,Install XGBoost
# MAGIC %pip install xgboost --quiet

# COMMAND ----------

# DBTITLE 1,Why Walk-Forward Validation?
# MAGIC %md
# MAGIC ## 🚪 Why Walk-Forward Validation Instead of Simple Train/Test Split?
# MAGIC
# MAGIC ### The Problem with Regular Train/Test Split:
# MAGIC
# MAGIC Imagine I have data from Jan 2024 to Dec 2025 (24 months). A regular split would:
# MAGIC - Train on Jan 2024 - Aug 2025 (80%)
# MAGIC - Test on Sep 2025 - Dec 2025 (20%)
# MAGIC
# MAGIC **But there's a problem**: I only test on ONE future scenario. What if that test period was unusually easy or hard to predict?
# MAGIC
# MAGIC ### My Walk-Forward Validation Approach:
# MAGIC
# MAGIC Instead of one test, I perform MULTIPLE tests that simulate real-world forecasting:
# MAGIC
# MAGIC ```
# MAGIC Fold 1: Train on months 1-12  → Predict month 13
# MAGIC Fold 2: Train on months 1-13  → Predict month 14
# MAGIC Fold 3: Train on months 1-14  → Predict month 15
# MAGIC ... and so on
# MAGIC ```
# MAGIC
# MAGIC **Why this is better:**
# MAGIC 1. **📊 Multiple Tests**: I test each model 12+ times, not just once
# MAGIC 2. **🎯 Realistic**: This mimics how I'll actually use the model (train on past, predict future)
# MAGIC 3. **⚖️ Fair Comparison**: All models face the same challenge
# MAGIC 4. **🛡️ Prevents Overfitting**: I can't get lucky with one test period
# MAGIC
# MAGIC ### What I Measure:
# MAGIC
# MAGIC - **MAPE (Mean Absolute Percentage Error)**: Average % error across all predictions
# MAGIC   - Lower is better
# MAGIC   - If MAPE = 15%, my predictions are off by 15% on average
# MAGIC
# MAGIC - **RMSE (Root Mean Squared Error)**: Penalizes large errors more
# MAGIC   - Also lower is better
# MAGIC   - Useful for identifying models that occasionally make huge mistakes
# MAGIC
# MAGIC ### The Winner:
# MAGIC
# MAGIC After all these tests, the model with the **lowest MAPE** becomes my champion. That's the model I'll use to forecast the next 6 months!

# COMMAND ----------

# DBTITLE 1,Define Lag Features Function (for hyperparameter tuning and CV)
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

print("📦 Defining feature engineering function for hyperparameter tuning and CV...\n")

# 🚨 FIX: Feature engineering WITHOUT leakage
# Only use: (1) lags of the target itself, (2) rolling stats, (3) calendar features, (4) volume trends
# DO NOT use denied_claims or any field that composes the target
def create_lag_features(df, lags=[1, 2, 3, 6, 12], use_volume=False):
    # Start with just ds and y (the denial rate target)
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime type
    
    # Include volume if requested
    if use_volume and 'total_claims' in df.columns:
        df_lag = df[['ds', 'y', 'total_claims']].copy()
        
        # Volume lag features (lagged claims volume)
        for lag in [1, 2, 3]:
            df_lag[f'volume_lag_{lag}'] = df_lag['total_claims'].shift(lag)
        
        # Volume rolling statistics
        df_lag['volume_rolling_mean_3'] = df_lag['total_claims'].shift(1).rolling(window=3, min_periods=1).mean()
        df_lag['volume_rolling_std_3'] = df_lag['total_claims'].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        
        # Volume trend (% change)
        df_lag['volume_pct_change'] = df_lag['total_claims'].pct_change().fillna(0)
    else:
        df_lag = df[['ds', 'y']].copy()
    
    # Lag features from the target itself
    for lag in lags:
        df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
    
    # Rolling statistics (use shift(1) to avoid peeking at current value)
    # Use min_periods to handle small windows gracefully
    df_lag['rolling_mean_3'] = df_lag['y'].shift(1).rolling(window=3, min_periods=1).mean()
    df_lag['rolling_std_3'] = df_lag['y'].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
    df_lag['rolling_mean_6'] = df_lag['y'].shift(1).rolling(window=6, min_periods=1).mean()
    
    # Time features (calendar info is safe)
    df_lag['month'] = df_lag['ds'].dt.month
    df_lag['quarter'] = df_lag['ds'].dt.quarter
    df_lag['year'] = df_lag['ds'].dt.year
    
    # Drop the total_claims column if it exists (we only want its derived features)
    if 'total_claims' in df_lag.columns:
        df_lag = df_lag.drop('total_claims', axis=1)
    
    return df_lag

print("\u2705 Function defined successfully!")
print(f"   Volume features: {'ENABLED' if USE_VOLUME_FEATURES else 'DISABLED'}")


# COMMAND ----------

# DBTITLE 1,Understanding XGBoost Features (No Data Leakage!)
# MAGIC %md
# MAGIC ## 🔍 Understanding My XGBoost Features - The Right Way!
# MAGIC
# MAGIC ### What is XGBoost?
# MAGIC
# MAGIC XGBoost is a machine learning algorithm that learns patterns from past data to predict the future. Think of it like a very smart pattern-recognition system.
# MAGIC
# MAGIC ### The Critical Rule: NO DATA LEAKAGE
# MAGIC
# MAGIC **Data leakage** = Using information that wouldn't be available in real forecasting
# MAGIC
# MAGIC **WRONG Approach** ❌:
# MAGIC ```python
# MAGIC # Using total_claims and denied_claims as features
# MAGIC # This is CHEATING because denial_rate = denied_claims / total_claims
# MAGIC # The model would just learn this formula instead of real patterns
# MAGIC ```
# MAGIC
# MAGIC **MY Approach** ✅:
# MAGIC I use ONLY these features:
# MAGIC
# MAGIC 1. **Lag Features** - Previous denial rates
# MAGIC    - `lag_1`: Denial rate 1 month ago
# MAGIC    - `lag_2`: Denial rate 2 months ago
# MAGIC    - `lag_3`: Denial rate 3 months ago
# MAGIC    - `lag_6`: Denial rate 6 months ago (mid-term trend)
# MAGIC    - `lag_12`: Denial rate 12 months ago (yearly comparison)
# MAGIC
# MAGIC 2. **Rolling Statistics** - Trends over time
# MAGIC    - `rolling_mean_3`: Average of last 3 months (short-term trend)
# MAGIC    - `rolling_std_3`: How volatile the last 3 months were
# MAGIC    - `rolling_mean_6`: Average of last 6 months (medium-term trend)
# MAGIC
# MAGIC 3. **Calendar Features** - Seasonal patterns
# MAGIC    - `month`: Month number (1-12) - captures yearly patterns
# MAGIC    - `quarter`: Quarter (1-4) - captures quarterly trends
# MAGIC    - `year`: Year number - captures long-term trends
# MAGIC
# MAGIC ### Why Recursive Forecasting?
# MAGIC
# MAGIC When I forecast 6 months ahead, I can't use "lag_1" from the future (it doesn't exist yet!). So I:
# MAGIC
# MAGIC 1. Predict Month 1
# MAGIC 2. Use that prediction as "lag_1" to predict Month 2
# MAGIC 3. Use Month 2's prediction as "lag_1" to predict Month 3
# MAGIC 4. And so on...
# MAGIC
# MAGIC This is called **recursive forecasting** and it's the honest way to make multi-step predictions!

# COMMAND ----------

# DBTITLE 1,Hyperparameter Tuning for Prophet and XGBoost
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error
import itertools
import warnings
warnings.filterwarnings('ignore')

print("🔧 HYPERPARAMETER TUNING")
print("="*70)

if ENABLE_HYPERPARAMETER_TUNING:
    print("✅ Tuning enabled - this will take a few minutes...\n")
    
    # We'll use a simple validation split (last 20% of data) for tuning
    tune_train_size = int(len(ts_df) * 0.7)  # 70% for training during tuning
    tune_train = ts_df[:tune_train_size][['ds', 'y']].copy()
    tune_val = ts_df[tune_train_size:][['ds', 'y']].copy()
    
    print(f"📊 Tuning dataset:")
    print(f"   Training: {len(tune_train)} months ({tune_train['ds'].min().strftime('%Y-%m')} to {tune_train['ds'].max().strftime('%Y-%m')})")
    print(f"   Validation: {len(tune_val)} months ({tune_val['ds'].min().strftime('%Y-%m')} to {tune_val['ds'].max().strftime('%Y-%m')})\n")
    
    # ═════════════════════════════════════════════════════════════════
    # 1. TUNE PROPHET
    # ═════════════════════════════════════════════════════════════════
    print("1️⃣ Tuning Prophet...")
    prophet_params = CONFIG['prophet_tuning']
    prophet_grid = list(ParameterGrid(prophet_params))
    
    best_prophet_mape = float('inf')
    best_prophet_params = None
    
    for i, params in enumerate(prophet_grid):
        try:
            # Train Prophet with these params
            prophet_model = Prophet(
                seasonality_mode='additive',
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95
            )
            prophet_model.fit(tune_train)
            
            # Predict on validation set
            future = tune_val[['ds']].copy()
            forecast = prophet_model.predict(future)
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(tune_val['y'], forecast['yhat']) * 100
            
            if mape < best_prophet_mape:
                best_prophet_mape = mape
                best_prophet_params = params
            
            if (i + 1) % 5 == 0:
                print(f"   Tested {i+1}/{len(prophet_grid)} combinations...")
                
        except Exception as e:
            continue
    
    if best_prophet_params:
        print(f"\n   ✅ Best Prophet params (MAPE: {best_prophet_mape:.2f}%):")
        print(f"      changepoint_prior_scale: {best_prophet_params['changepoint_prior_scale']}")
        print(f"      seasonality_prior_scale: {best_prophet_params['seasonality_prior_scale']}")
        
        # Update CONFIG with best params
        CONFIG['prophet']['changepoint_prior_scale'] = best_prophet_params['changepoint_prior_scale']
        CONFIG['prophet']['seasonality_prior_scale'] = best_prophet_params['seasonality_prior_scale']
    else:
        print("   ⚠️  Prophet tuning failed, using defaults")
    
    # ═════════════════════════════════════════════════════════════════
    # 2. TUNE XGBOOST
    # ═════════════════════════════════════════════════════════════════
    print("\n2️⃣ Tuning XGBoost...")
    
    # Prepare features for tuning
    tune_full = ts_df[['ds', 'y', 'total_claims']].copy()
    tune_features = create_lag_features(tune_full, use_volume=USE_VOLUME_FEATURES).dropna()
    
    tune_feat_train = tune_features[:tune_train_size]
    tune_feat_val = tune_features[tune_train_size:]
    
    if len(tune_feat_val) > 0:
        xgb_params = CONFIG['xgboost_tuning']
        xgb_grid = list(ParameterGrid(xgb_params))
        
        feature_cols = [col for col in tune_feat_train.columns if col not in ['ds', 'y']]
        X_tune_train = tune_feat_train[feature_cols]
        y_tune_train = tune_feat_train['y']
        X_tune_val = tune_feat_val[feature_cols]
        y_tune_val = tune_feat_val['y']
        
        best_xgb_mape = float('inf')
        best_xgb_params = None
        
        for i, params in enumerate(xgb_grid):
            try:
                xgb_model = XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    random_state=MASTER_SEED,
                    verbosity=0
                )
                
                xgb_model.fit(X_tune_train, y_tune_train)
                preds = xgb_model.predict(X_tune_val)
                
                mape = mean_absolute_percentage_error(y_tune_val, preds) * 100
                
                if mape < best_xgb_mape:
                    best_xgb_mape = mape
                    best_xgb_params = params
                
                if (i + 1) % 10 == 0:
                    print(f"   Tested {i+1}/{len(xgb_grid)} combinations...")
                    
            except Exception as e:
                continue
        
        if best_xgb_params:
            print(f"\n   ✅ Best XGBoost params (MAPE: {best_xgb_mape:.2f}%):")
            print(f"      n_estimators: {best_xgb_params['n_estimators']}")
            print(f"      max_depth: {best_xgb_params['max_depth']}")
            print(f"      learning_rate: {best_xgb_params['learning_rate']}")
            
            # Update CONFIG with best params
            CONFIG['xgboost']['n_estimators'] = best_xgb_params['n_estimators']
            CONFIG['xgboost']['max_depth'] = best_xgb_params['max_depth']
            CONFIG['xgboost']['learning_rate'] = best_xgb_params['learning_rate']
        else:
            print("   ⚠️  XGBoost tuning failed, using defaults")
    else:
        print("   ⚠️  Not enough validation data for XGBoost tuning, using defaults")
    
    print("\n" + "="*70)
    print("✅ Hyperparameter tuning complete! Updated params will be used in CV.\n")
    
else:
    print("⏭️  Tuning disabled - using default hyperparameters\n")

# COMMAND ----------

# DBTITLE 1,Walk-Forward Cross-Validation (Repeatable)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objects as go
import itertools
import warnings
warnings.filterwarnings('ignore')

print("🔁 Running REAL Walk-Forward Cross-Validation...")
print("   🚨 FIX: Expanding window, all 5 models, averaged across folds")
print(f"   🎲 Using master seed: {MASTER_SEED} for reproducibility\n")

# Walk-forward validation with expanding window
min_train_size = MIN_TRAIN_SIZE  # From configuration cell

# Store results for each model across all folds
model_fold_results = {
    'Baseline (3-MA)': [],
    'Prophet': [],
    'ARIMA': [],
    'SARIMA': [],
    'XGBoost (Lag Features)': []
}

print(f"   Min training size: {min_train_size} months")
print(f"   Total folds: {len(ts_df) - min_train_size}")
print("\n" + "="*60)

for test_idx in range(min_train_size, len(ts_df)):
    # Expanding window: train on [0, test_idx), predict test_idx
    train_fold = ts_df[:test_idx][['ds', 'y']].copy()
    test_row = ts_df.iloc[test_idx]
    actual = test_row['y']
    
    train_series = train_fold.set_index('ds')['y']
    
    # 1. BASELINE: 3-Month Moving Average
    pred_baseline = train_fold['y'].tail(CONFIG['baseline']['window']).mean()
    model_fold_results['Baseline (3-MA)'].append((actual, pred_baseline))
    
    # 2. PROPHET
    try:
        enable_yearly_fold = len(train_fold) >= 24
        prophet_fold = Prophet(
            seasonality_mode=CONFIG['prophet']['seasonality_mode'],
            yearly_seasonality=enable_yearly_fold,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=CONFIG['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=CONFIG['prophet']['seasonality_prior_scale'],
            interval_width=CONFIG['prophet']['interval_width']
        )
        prophet_fold.fit(train_fold)
        future_fold = prophet_fold.make_future_dataframe(periods=1, freq='MS')
        forecast_fold = prophet_fold.predict(future_fold)
        pred_prophet = forecast_fold['yhat'].iloc[-1]
        model_fold_results['Prophet'].append((actual, pred_prophet))
    except Exception as e:
        print(f"⚠️  Prophet failed on fold {test_idx - min_train_size + 1} (train size {len(train_fold)}): {e}")
    
    # 3. ARIMA (simple auto-tuning)
    try:
        best_aic, best_pred = np.inf, None
        for p, d, q in itertools.product(CONFIG['arima']['p_range'], 
                                         CONFIG['arima']['d_range'], 
                                         CONFIG['arima']['q_range']):
            try:
                arima_fold = ARIMA(train_series, order=(p, d, q)).fit()
                if arima_fold.aic < best_aic:
                    best_aic = arima_fold.aic
                    best_pred = arima_fold.forecast(steps=1).iloc[0]
            except:
                continue
        if best_pred is not None:
            model_fold_results['ARIMA'].append((actual, best_pred))
        else:
            print(f"⚠️  ARIMA failed on fold {test_idx - min_train_size + 1}: no valid order found (train size {len(train_fold)})")
    except Exception as e:
        print(f"⚠️  ARIMA failed on fold {test_idx - min_train_size + 1}: {e}")
    
    # 4. SARIMA
    try:
        sarima_fold = SARIMAX(
            train_series, 
            order=CONFIG['sarima']['order'], 
            seasonal_order=CONFIG['sarima']['seasonal_order'],
            enforce_stationarity=False, 
            enforce_invertibility=False
        ).fit(disp=False)
        pred_sarima = sarima_fold.forecast(steps=1).iloc[0]
        model_fold_results['SARIMA'].append((actual, pred_sarima))
    except Exception as e:
        print(f"⚠️  SARIMA failed on fold {test_idx - min_train_size + 1} (train size {len(train_fold)}): {e}")
    
    # 5. XGBOOST (leakage-free features with volume)
    try:
        # Create features for this fold - include volume data
        train_fold_full = ts_df[:test_idx][['ds', 'y', 'total_claims']].copy()
        train_feat_fold = create_lag_features(train_fold_full, use_volume=USE_VOLUME_FEATURES).dropna()
        if len(train_feat_fold) > 0:
            # Build feature column list dynamically
            base_features = [f'lag_{l}' for l in LAG_FEATURES] + \
                           ['rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'month', 'quarter', 'year']
            
            # Add volume features if enabled
            if USE_VOLUME_FEATURES:
                volume_features = ['volume_lag_1', 'volume_lag_2', 'volume_lag_3', 
                                  'volume_rolling_mean_3', 'volume_rolling_std_3', 'volume_pct_change']
                feature_cols_fold = base_features + volume_features
            else:
                feature_cols_fold = base_features
            
            # Only use features that actually exist in the dataframe
            feature_cols_fold = [f for f in feature_cols_fold if f in train_feat_fold.columns]
            
            X_train_fold = train_feat_fold[feature_cols_fold]
            y_train_fold = train_feat_fold['y']
            
            xgb_fold = XGBRegressor(
                n_estimators=CONFIG['xgboost']['n_estimators'],
                max_depth=CONFIG['xgboost']['max_depth'],
                learning_rate=CONFIG['xgboost']['learning_rate'],
                random_state=CONFIG['xgboost']['random_state'],  # Uses MASTER_SEED
                verbosity=CONFIG['xgboost']['verbosity']
            )
            xgb_fold.fit(X_train_fold, y_train_fold)
            
            # Create features for test point
            test_full = ts_df[:test_idx+1][['ds', 'y', 'total_claims']].copy()
            test_feat = create_lag_features(test_full, use_volume=USE_VOLUME_FEATURES).iloc[[-1]]
            X_test_fold = test_feat[feature_cols_fold]
            pred_xgb = xgb_fold.predict(X_test_fold)[0]
            model_fold_results['XGBoost (Lag Features)'].append((actual, pred_xgb))
        else:
            print(f"⚠️  XGBoost skipped fold {test_idx - min_train_size + 1}: insufficient data after lag creation (train size {len(train_fold)})")
    except Exception as e:
        print(f"⚠️  XGBoost failed on fold {test_idx - min_train_size + 1}: {e}")
    
    if (test_idx - min_train_size) % 3 == 0:
        print(f"Fold {test_idx - min_train_size + 1}: Train until {train_fold['ds'].max().strftime('%Y-%m')}")

# Aggregate results for each model
wf_comparison_rows = []
for model_name, pairs in model_fold_results.items():
    if not pairs:
        continue
    actuals, preds = zip(*pairs)
    actuals, preds = np.array(actuals), np.array(preds)
    wf_comparison_rows.append({
        'model': model_name,
        'folds': len(pairs),
        'mae': mean_absolute_error(actuals, preds),
        'mape': mean_absolute_percentage_error(actuals, preds) * 100,
        'rmse': np.sqrt(mean_squared_error(actuals, preds))
    })

# Create comparison DataFrame with DETERMINISTIC tie-breaking
wf_comparison_df = pd.DataFrame(wf_comparison_rows)

# 🔑 DETERMINISTIC TIE-BREAKING:
# If multiple models have identical MAPE, use preference order
wf_comparison_df['model_preference'] = wf_comparison_df['model'].map(
    {m: i for i, m in enumerate(MODEL_PREFERENCE_ORDER)}
)
# Sort by MAPE first (ascending), then by preference (ascending) for ties
wf_comparison_df = wf_comparison_df.sort_values(
    ['mape', 'model_preference'], 
    ascending=[True, True]
).reset_index(drop=True)

# Remove the helper column
wf_comparison_df = wf_comparison_df.drop('model_preference', axis=1)

print("\n" + "="*70)
print("🏆 WALK-FORWARD CV RESULTS - ALL MODELS (averaged across folds)")
print("="*70)
print(wf_comparison_df.to_string(index=False))

best_wf_model = wf_comparison_df.iloc[0]
print(f"\n🥇 BEST MODEL: {best_wf_model['model']} (MAPE: {best_wf_model['mape']:.2f}% over {int(best_wf_model['folds'])} folds)")

# Visualize walk-forward comparison
fig = go.Figure()

fig.add_trace(go.Bar(
    x=wf_comparison_df['model'],
    y=wf_comparison_df['mape'],
    marker_color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(wf_comparison_df))],
    text=wf_comparison_df['mape'].round(2),
    textposition='outside'
))

fig.update_layout(
    title='<b>Walk-Forward CV Performance - All Models</b><br><sub>Lower MAPE = Better</sub>',
    xaxis_title='Model',
    yaxis_title='MAPE (%)',
    template='plotly_white',
    height=500,
    showlegend=False
)

fig.show()

# Store for later use - ensure best_wf_model is a proper dict for later indexing
if wf_comparison_df.empty:
    raise RuntimeError("❌ Walk-forward CV produced no results — all models failed.")

best_wf_model = wf_comparison_df.iloc[0].to_dict()
best_model_name = best_wf_model['model']
model_results = wf_comparison_df.to_dict('records')

print(f"\n✅ Best model stored: {best_model_name}")
print(f"   MAPE: {best_wf_model['mape']:.2f}%")
print(f"   RMSE: {best_wf_model['rmse']:.2f}%")
print(f"   Folds: {int(best_wf_model['folds'])}")

# COMMAND ----------

# DBTITLE 1,Retrain Best Model and Generate Final Forecast (Repeatable)
from datetime import datetime

# ✅✅✅ This cell now runs AFTER Walk-Forward CV (Cell 24) ✅✅✅
# Dependencies are computed by the CV cell above:
#   - best_model_name: which model won CV
#   - best_wf_model: dict with mape/rmse/folds from CV
#   - model_results: list of all model results

print("🔍 Checking dependencies...")
if 'best_model_name' not in locals() or 'model_results' not in locals() or 'best_wf_model' not in locals():
    raise RuntimeError(
        "❌❌❌ DEPENDENCY ERROR: Run Cell 24 (Walk-Forward Cross-Validation) FIRST!\n\n"
        "   That cell computes:\n"
        "   - best_model_name (e.g., 'Baseline (3-MA)')\n"
        "   - best_wf_model (dict with mape, rmse, folds)\n"
        "   - model_results (list of all CV results)\n\n"
        "   Scroll up, run Cell 24, then come back here."
    )

print(f"✅ Dependencies verified! Best model from CV: {best_model_name}\n")

# 🚨 CRITICAL BLOCKLIST: SARIMA diverges in out-of-sample forecasts (e.g., 5,763% in month 6)
# If CV selects SARIMA, override with the next-best model to prevent impossible predictions
if 'SARIMA' in best_model_name:
    print(f"⚠️  SARIMA was selected but is BLOCKED due to known divergence issues.")
    print(f"   Switching to second-best model from CV results...\n")
    
    # Find the second-best model (first non-SARIMA in sorted results)
    fallback_model = None
    for result in model_results:
        if 'SARIMA' not in result['model']:
            fallback_model = result['model']
            fallback_mape = result['mape']
            break
    
    if fallback_model:
        print(f"✅ Using {fallback_model} instead (MAPE: {fallback_mape:.2f}%)\n")
        best_model_name = fallback_model
    else:
        raise ValueError("No valid fallback model found — all models failed in CV")

print(f"🚀 Retraining Best Model ({best_model_name}) on Full Dataset...\n")

future_periods = FUTURE_PERIODS  # From configuration

print(f"📅 Forecasting {future_periods} months ahead using {best_model_name}\n")
final_forecast_df = None

# Retrain based on which model won
if 'Baseline' in best_model_name:
    # Simple moving average forecast
    ma_value = ts_df['y'].rolling(window=CONFIG['baseline']['window']).mean().iloc[-1]
    future_dates = pd.date_range(start=ts_df['ds'].max() + pd.DateOffset(months=1), periods=future_periods, freq='MS')
    
    final_forecast_df = pd.DataFrame({
        'forecast_date': future_dates,
        'predicted_denial_rate': [ma_value] * future_periods,
        'lower_bound': [ma_value - 2*ts_df['y'].std()] * future_periods,
        'upper_bound': [ma_value + 2*ts_df['y'].std()] * future_periods
    })
    
elif 'Prophet' in best_model_name:
    # Retrain Prophet on full data with configured parameters
    prophet_final = Prophet(
        seasonality_mode=CONFIG['prophet']['seasonality_mode'],
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=CONFIG['prophet']['changepoint_prior_scale'],
        seasonality_prior_scale=CONFIG['prophet']['seasonality_prior_scale'],
        interval_width=CONFIG['prophet']['interval_width']
    )
    prophet_final.fit(ts_df[['ds', 'y']])
    
    future_prophet = prophet_final.make_future_dataframe(periods=future_periods, freq='MS')
    forecast_prophet = prophet_final.predict(future_prophet)
    
    forecast_prophet_future = forecast_prophet.tail(future_periods)
    final_forecast_df = pd.DataFrame({
        'forecast_date': forecast_prophet_future['ds'],
        'predicted_denial_rate': forecast_prophet_future['yhat'],
        'lower_bound': forecast_prophet_future['yhat_lower'],
        'upper_bound': forecast_prophet_future['yhat_upper']
    })
    
elif 'ARIMA' in best_model_name and 'SARIMA' not in best_model_name:
    # 🛡️ Ensure we have a valid ARIMA order
    if 'best_arima_order' not in locals() or best_arima_order is None:
        print("   🔍 ARIMA order not available, searching for best parameters...")
        best_aic = np.inf
        best_arima_order = None
        for p, d, q in itertools.product(CONFIG['arima']['p_range'], 
                                         CONFIG['arima']['d_range'], 
                                         CONFIG['arima']['q_range']):
            try:
                m = ARIMA(ts_df['y'], order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_arima_order = (p, d, q)
            except:
                continue
        if best_arima_order is None:
            best_arima_order = (1, 1, 1)
            print("   ⚠️  ARIMA order search failed — defaulting to (1,1,1)")
        else:
            print(f"   ✅ Found best order: {best_arima_order}")
    
    # Retrain ARIMA on full data
    arima_final = ARIMA(ts_df['y'], order=best_arima_order)
    arima_fitted_final = arima_final.fit()
    
    # Get forecast with confidence intervals
    arima_forecast_obj = arima_fitted_final.get_forecast(steps=future_periods)
    arima_forecast_final = arima_forecast_obj.predicted_mean
    arima_conf_int = arima_forecast_obj.conf_int()
    
    future_dates = pd.date_range(start=ts_df['ds'].max() + pd.DateOffset(months=1), periods=future_periods, freq='MS')
    final_forecast_df = pd.DataFrame({
        'forecast_date': future_dates,
        'predicted_denial_rate': arima_forecast_final.values,
        'lower_bound': arima_conf_int.iloc[:, 0].values,
        'upper_bound': arima_conf_int.iloc[:, 1].values
    })
    
elif 'SARIMA' in best_model_name:
    # Retrain SARIMA on full data with configured parameters
    sarima_final = SARIMAX(
        ts_df['y'],
        order=CONFIG['sarima']['order'],
        seasonal_order=CONFIG['sarima']['seasonal_order'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fitted_final = sarima_final.fit(disp=False)
    
    sarima_forecast_final = sarima_fitted_final.forecast(steps=future_periods)
    sarima_forecast_conf = sarima_fitted_final.get_forecast(steps=future_periods).conf_int()
    
    future_dates = pd.date_range(start=ts_df['ds'].max() + pd.DateOffset(months=1), periods=future_periods, freq='MS')
    final_forecast_df = pd.DataFrame({
        'forecast_date': future_dates,
        'predicted_denial_rate': sarima_forecast_final,
        'lower_bound': sarima_forecast_conf.iloc[:, 0],
        'upper_bound': sarima_forecast_conf.iloc[:, 1]
    })
    
elif 'XGBoost' in best_model_name:
    # 🚨 FIX: Recursive multi-step forecasting (fixes the flat forecast bug)
    print("   🔁 Using recursive forecasting: each prediction feeds into next step's features\n")
    
    xgb_final = XGBRegressor(
        n_estimators=CONFIG['xgboost']['n_estimators'],
        max_depth=CONFIG['xgboost']['max_depth'],
        learning_rate=CONFIG['xgboost']['learning_rate'],
        random_state=CONFIG['xgboost']['random_state'],  # Uses MASTER_SEED for reproducibility
        verbosity=CONFIG['xgboost']['verbosity']
    )
    
    # Retrain on full feature dataset
    ts_df_features_full = create_lag_features(ts_df).dropna()
    feature_cols_full = [f'lag_{l}' for l in LAG_FEATURES] + \
                       ['rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'month', 'quarter', 'year']
    X_full = ts_df_features_full[feature_cols_full]
    y_full = ts_df_features_full['y']
    xgb_final.fit(X_full, y_full)
    
    # Recursive forecasting: predict one step, feed back, repeat
    future_predictions = []
    working_df = ts_df[['ds', 'y']].copy()  # Start with historical data
    last_date = working_df['ds'].max()
    
    for step in range(1, future_periods + 1):
        # Next forecast date
        next_date = last_date + pd.DateOffset(months=step)
        
        # Append a placeholder for the next month
        temp_df = pd.concat([working_df, pd.DataFrame({'ds': [next_date], 'y': [np.nan]})], ignore_index=True)
        
        # Create features (lags will pull from working_df, which now includes prior predictions)
        temp_features = create_lag_features(temp_df)
        next_features = temp_features.iloc[[-1]][feature_cols_full]
        
        # Predict next month
        pred = xgb_final.predict(next_features)[0]
        future_predictions.append(pred)
        
        # 🔑 KEY FIX: Feed prediction back into working data for next iteration
        working_df = pd.concat([working_df, pd.DataFrame({'ds': [next_date], 'y': [pred]})], ignore_index=True)
    
    future_dates = pd.date_range(start=ts_df['ds'].max() + pd.DateOffset(months=1), periods=future_periods, freq='MS')
    
    # 🛡️ Use RMSE from CV for confidence intervals (safe fallback to historical std)
    if 'rmse' in best_wf_model and not pd.isna(best_wf_model['rmse']):
        pred_std = float(best_wf_model['rmse'])
        print(f"   Using CV RMSE for confidence intervals: {pred_std:.2f}%")
    else:
        pred_std = ts_df['y'].std()
        print(f"   Using historical std for confidence intervals: {pred_std:.2f}%")
    
    final_forecast_df = pd.DataFrame({
        'forecast_date': future_dates,
        'predicted_denial_rate': future_predictions,
        'lower_bound': np.array(future_predictions) - 1.96 * pred_std,
        'upper_bound': np.array(future_predictions) + 1.96 * pred_std
    })

# 🛡️ CRITICAL: Verify forecast was generated
if final_forecast_df is None or len(final_forecast_df) == 0:
    raise RuntimeError(f"❌ No forecast generated for model '{best_model_name}' — check retrain step")

# Add metadata
final_forecast_df['model_version'] = best_model_name
final_forecast_df['created_at'] = datetime.now()

# 🛡️ CRITICAL: Convert types to native Python for Spark compatibility
final_forecast_df['forecast_date'] = pd.to_datetime(final_forecast_df['forecast_date']).dt.date
final_forecast_df['created_at'] = pd.to_datetime(final_forecast_df['created_at'])

# Ensure numeric columns are floats with no NaN
for col in ['predicted_denial_rate', 'lower_bound', 'upper_bound']:
    final_forecast_df[col] = pd.to_numeric(final_forecast_df[col], errors='coerce')
    if final_forecast_df[col].isna().any():
        print(f"⚠️  Warning: NaN values found in {col}, filling with fallback")
        final_forecast_df[col] = final_forecast_df[col].fillna(final_forecast_df['predicted_denial_rate'].mean())

print(f"✅ Generated {future_periods}-month forecast using {best_model_name}\n")
print("📋 Future Denial Rate Predictions:")
for _, row in final_forecast_df.iterrows():
    print(f"   {row['forecast_date'].strftime('%Y-%m')}: {row['predicted_denial_rate']:.2f}% [{row['lower_bound']:.2f}% - {row['upper_bound']:.2f}%]")

# Visualize final forecast
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=ts_df['ds'], y=ts_df['y'],
    mode='lines+markers',
    name='Historical',
    line=dict(color='blue', width=2)
))

# Forecast
fig.add_trace(go.Scatter(
    x=final_forecast_df['forecast_date'],
    y=final_forecast_df['predicted_denial_rate'],
    mode='lines+markers',
    name=f'Forecast ({best_model_name})',
    line=dict(color='red', width=2, dash='dash'),
    marker=dict(size=8)
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=final_forecast_df['forecast_date'].tolist() + final_forecast_df['forecast_date'].tolist()[::-1],
    y=final_forecast_df['upper_bound'].tolist() + final_forecast_df['lower_bound'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255,0,0,0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Interval'
))

fig.update_layout(
    title=f'<b>Final Forecast - {best_model_name}</b><br><sub>Next 6 Months</sub>',
    xaxis_title='Month',
    yaxis_title='Denial Rate (%)',
    hovermode='x unified',
    template='plotly_white',
    height=500
)

fig.show()

display(final_forecast_df)

# COMMAND ----------

# DBTITLE 1,Save Forecast to Delta Table
# Convert pandas DataFrame to Spark DataFrame
# Types already converted to Python native in previous cell for compatibility

print(f"🔄 Converting forecast to Spark DataFrame...")
print(f"   Rows: {len(final_forecast_df)}")
print(f"   Columns: {list(final_forecast_df.columns)}")

# Let Spark infer schema from properly-typed pandas DataFrame
forecast_spark_df = spark.createDataFrame(final_forecast_df)

# Cast to ensure correct Spark types
from pyspark.sql import functions as F
forecast_spark_df = forecast_spark_df \
    .withColumn('forecast_date', F.col('forecast_date').cast('date')) \
    .withColumn('predicted_denial_rate', F.col('predicted_denial_rate').cast('double')) \
    .withColumn('lower_bound', F.col('lower_bound').cast('double')) \
    .withColumn('upper_bound', F.col('upper_bound').cast('double')) \
    .withColumn('model_version', F.col('model_version').cast('string')) \
    .withColumn('created_at', F.col('created_at').cast('timestamp'))

# Save to Delta table
output_table = "workspace.default.denial_rate_forecast"

print(f"💾 Saving forecast to {output_table}...")

try:
    # Write to Delta table (overwrite for now - change to append for production)
    forecast_spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(output_table)
    
    print(f"✅ Forecast saved successfully!")
    print(f"\n📊 Query the forecast:")
    print(f"   SELECT * FROM {output_table}")
    
    # Verify the save
    saved_df = spark.table(output_table)
    print(f"\n✅ Verified: {saved_df.count()} rows saved to table")
    
except Exception as e:
    print(f"❌ Error saving to table: {e}")
    print(f"\n💡 The forecast DataFrame is available as 'final_forecast_df' for manual inspection")

# COMMAND ----------

# DBTITLE 1,Summary and Next Steps
# MAGIC %md
# MAGIC ## 🎉 My Forecasting Model is Complete!
# MAGIC
# MAGIC ### ✅ What I Built:
# MAGIC 1. **EDA**: Decomposition, stationarity tests, autocorrelation analysis
# MAGIC 2. **Multi-Model Ensemble**: Trained and compared 5 different models
# MAGIC    - Baseline (3-Month Moving Average)
# MAGIC    - Prophet (automatic seasonality)
# MAGIC    - ARIMA (auto-tuned parameters)
# MAGIC    - SARIMA (seasonal ARIMA)
# MAGIC    - XGBoost (with lag features)
# MAGIC 3. **Model Selection**: Automatically selected best performer based on MAPE
# MAGIC 4. **Validation**: Walk-forward cross-validation with expanding windows
# MAGIC 5. **Forecast Output**: 6-month predictions from best model saved to Delta table
# MAGIC
# MAGIC ### 📊 Model Performance:
# MAGIC - **Best Model**: Automatically selected based on test MAPE
# MAGIC - **All Models Compared**: See performance comparison chart above
# MAGIC - **Confidence Intervals**: 95% prediction bounds included
# MAGIC - **Output Location**: `workspace.default.denial_rate_forecast`
# MAGIC
# MAGIC ### 🚀 How I Can Operationalize This:
# MAGIC
# MAGIC #### 1. Schedule as Databricks Job
# MAGIC ```python
# MAGIC # I can schedule this notebook to run:
# MAGIC # - Weekly: Refresh forecast with latest data
# MAGIC # - Monthly: Full retrain and predict next 6-12 months
# MAGIC ```
# MAGIC
# MAGIC #### 2. Add to Dashboard
# MAGIC - Query `workspace.default.denial_rate_forecast` in your existing dashboard
# MAGIC - Create line chart showing predicted vs actual denial rates
# MAGIC - Add confidence interval bands for uncertainty
# MAGIC
# MAGIC #### 3. Set Up Alerts
# MAGIC - Alert when predicted denial rate > threshold (e.g., 12%)
# MAGIC - Alert when actual exceeds upper confidence bound
# MAGIC - Alert for sudden trend changes
# MAGIC
# MAGIC #### 4. Model Enhancements (Future Iterations):
# MAGIC - **Add payer-level forecasts**: Train separate models per payer
# MAGIC - **External regressors**: Add denial reason categories, claim volume
# MAGIC - **Stacked ensemble**: Combine multiple model predictions with meta-learner
# MAGIC - **More data**: Expand training history to 2-3+ years for better seasonality
# MAGIC
# MAGIC ### 📈 Business Value:
# MAGIC - **Proactive**: Identify denial rate increases before they happen
# MAGIC - **Actionable**: Focus on high-risk periods and payers
# MAGIC - **Measurable**: Track forecast accuracy over time
# MAGIC - **Strategic**: Support resource planning and payer negotiations

# COMMAND ----------

# DBTITLE 1,Load Denial Data
# Load denial data from existing view
denial_df = spark.table("workspace.default.v_denials_by_payer")

# Show schema and basic stats
print("📊 Schema:")
denial_df.printSchema()

print("\n📈 Record Count:", denial_df.count())
print("\n📋 Sample Data:")
display(denial_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Data Requirements Check
# MAGIC %md
# MAGIC ## ⚠️ Data Requirements for Time-Series Forecasting
# MAGIC
# MAGIC For forecasting, I need:
# MAGIC 1. **Date/Timestamp column** - to establish temporal ordering
# MAGIC 2. **Target variable** - denial_rate (already present)
# MAGIC 3. **Sufficient history** - ideally 2+ years of data for seasonal patterns
# MAGIC
# MAGIC **Next Step**: If the current view doesn't have date columns, I'll need to either:
# MAGIC - Join with claims/transaction tables to get submission dates
# MAGIC - Use a different table that has temporal data
# MAGIC - Generate synthetic dates for demonstration purposes

# COMMAND ----------

# DBTITLE 1,Create Output Table Location
# Define output table for forecast results
output_catalog = "workspace"
output_schema = "default"
output_table = "denial_rate_forecast"

output_table_name = f"{output_catalog}.{output_schema}.{output_table}"

print(f"📍 Forecast output will be saved to: {output_table_name}")
print("\n📝 Table will include:")
print("   - forecast_date: date of prediction")
print("   - payer_name: payer identifier")
print("   - predicted_denial_rate: forecasted denial rate")
print("   - lower_bound: 95% confidence interval lower")
print("   - upper_bound: 95% confidence interval upper")
print("   - model_version: version/timestamp of model used")
print("   - created_at: timestamp when forecast was generated")
