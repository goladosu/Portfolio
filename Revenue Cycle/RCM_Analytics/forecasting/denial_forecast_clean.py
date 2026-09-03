
# RCM Denial Rate Forecasting
# ============================
# 
# The whole point: run several models through the same honest test (walk-forward 
# validation), let the numbers pick a winner, then use THAT winner to forecast.

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings("ignore")

# Settings
CLAIMS_PATH = "/Workspace/Users/goladosurahman@gmail.com/Portfolio/Revenue Cycle/dataset/fact_claims.csv"
OUTPUT_TABLE = "workspace.default.denial_rate_forecast"
FORECAST_HORIZON = 6
MIN_TRAIN_MONTHS = 12
PARTIAL_MONTH_VOLUME_RATIO = 0.5
LAGS = [1, 2, 3, 6, 12]

def load_monthly_denial_rate(path):
    """Read claims data and turn it into one row per month."""
    try:
        claims = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path).toPandas()
    except NameError:
        claims = pd.read_csv(path)
    
    claims["submission_date"] = pd.to_datetime(claims["submission_date"])
    
    monthly = (
        claims.assign(month=claims["submission_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(total_claims=("claim_id", "count"), denied_claims=("denial_flag", "sum"))
        .reset_index()
        .rename(columns={"month": "ds"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    monthly["y"] = monthly["denied_claims"] / monthly["total_claims"] * 100
    
    # Drop partial month
    median_volume = monthly["total_claims"].iloc[:-1].median()
    if monthly["total_claims"].iloc[-1] < PARTIAL_MONTH_VOLUME_RATIO * median_volume:
        print(f"Dropping {monthly['ds'].iloc[-1].strftime('%Y-%m')} — incomplete")
        monthly = monthly.iloc[:-1].reset_index(drop=True)
    
    return monthly[["ds", "y"]]

# Model classes - every one has fit() and predict()

class MovingAverage:
    name = "Baseline (3-month average)"
    
    def __init__(self, window=3):
        self.window = window
    
    def fit(self, history):
        self.recent_values = list(history["y"])
        return self
    
    def predict(self, n_steps):
        values = list(self.recent_values)
        forecasts = []
        for _ in range(n_steps):
            next_value = np.mean(values[-self.window:])
            forecasts.append(next_value)
            values.append(next_value)
        return np.array(forecasts)

class AutoArima:
    name = "ARIMA"
    
    def fit(self, history):
        from statsmodels.tsa.arima.model import ARIMA
        series = history.set_index("ds")["y"]
        
        best_fit, best_aic = None, np.inf
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        candidate = ARIMA(series, order=(p, d, q)).fit()
                        if candidate.aic < best_aic:
                            best_fit, best_aic = candidate, candidate.aic
                    except Exception:
                        continue
        
        if best_fit is None:
            raise RuntimeError("no ARIMA order converged")
        self.fitted = best_fit
        return self
    
    def predict(self, n_steps):
        return self.fitted.forecast(steps=n_steps).values

class SeasonalArima:
    name = "SARIMA"
    ORDERS_TO_TRY = [
        ((1, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 0), (1, 0, 0, 12)),
        ((0, 1, 1), (0, 1, 0, 12)),
    ]
    
    def fit(self, history):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        series = history.set_index("ds")["y"]
        
        best_fit, best_aic = None, np.inf
        for order, seasonal_order in self.ORDERS_TO_TRY:
            try:
                candidate = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if candidate.aic < best_aic:
                    best_fit, best_aic = candidate, candidate.aic
            except Exception:
                continue
        
        if best_fit is None:
            raise RuntimeError("no SARIMA configuration converged")
        self.fitted = best_fit
        return self
    
    def predict(self, n_steps):
        return self.fitted.forecast(steps=n_steps).values

class ProphetModel:
    name = "Prophet"
    
    def fit(self, history):
        from prophet import Prophet
        enough_history = len(history) >= 24
        
        model = Prophet(
            yearly_seasonality=enough_history,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(history[["ds", "y"]])
        self.model = model
        return self
    
    def predict(self, n_steps):
        future = self.model.make_future_dataframe(periods=n_steps, freq="MS")
        return self.model.predict(future)["yhat"].tail(n_steps).values

class XgboostOnLagFeatures:
    name = "XGBoost"
    
    def __init__(self, lags):
        self.lags = lags
    
    @property
    def feature_columns(self):
        return [f"lag_{lag}" for lag in self.lags] + [
            "rolling_mean_3", "rolling_std_3", "rolling_mean_6", 
            "month", "quarter", "year"
        ]
    
    def _add_features(self, df):
        out = df[["ds", "y"]].copy()
        for lag in self.lags:
            out[f"lag_{lag}"] = out["y"].shift(lag)
        out["rolling_mean_3"] = out["y"].shift(1).rolling(3).mean()
        out["rolling_std_3"] = out["y"].shift(1).rolling(3).std()
        out["rolling_mean_6"] = out["y"].shift(1).rolling(6).mean()
        out["month"] = out["ds"].dt.month
        out["quarter"] = out["ds"].dt.quarter
        out["year"] = out["ds"].dt.year
        return out
    
    def fit(self, history):
        from xgboost import XGBRegressor
        featured = self._add_features(history).dropna()
        if featured.empty:
            raise RuntimeError("not enough history for lag features")
        
        self.model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                                 random_state=42, verbosity=0)
        self.model.fit(featured[self.feature_columns], featured["y"])
        self.known_history = history[["ds", "y"]].copy()
        return self
    
    def predict(self, n_steps):
        working_history = self.known_history.copy()
        forecasts = []
        last_date = working_history["ds"].max()
        
        for step in range(1, n_steps + 1):
            next_date = last_date + pd.DateOffset(months=step)
            preview = pd.concat([working_history, 
                               pd.DataFrame({"ds": [next_date], "y": [np.nan]})], 
                              ignore_index=True)
            next_row = self._add_features(preview).iloc[[-1]][self.feature_columns].astype(float)
            
            prediction = self.model.predict(next_row)[0]
            forecasts.append(prediction)
            working_history = pd.concat([working_history, 
                                        pd.DataFrame({"ds": [next_date], "y": [prediction]})], 
                                       ignore_index=True)
        
        return np.array(forecasts)

def walk_forward_validate(monthly, models, min_train_months=MIN_TRAIN_MONTHS):
    """Test every model with walk-forward cross-validation."""
    scores = {model.name: [] for model in models}
    
    for cutoff in range(min_train_months, len(monthly)):
        train_history = monthly.iloc[:cutoff].reset_index(drop=True)
        actual_next = monthly["y"].iloc[cutoff]
        
        for model in models:
            try:
                model.fit(train_history)
                predicted_next = model.predict(1)[0]
                scores[model.name].append((actual_next, predicted_next))
            except Exception as error:
                pass  # Model sat this fold out
    
    leaderboard = []
    for model_name, pairs in scores.items():
        if not pairs:
            continue
        actuals, predictions = zip(*pairs)
        actuals, predictions = np.array(actuals), np.array(predictions)
        leaderboard.append({
            "model": model_name,
            "months_tested": len(pairs),
            "mae": mean_absolute_error(actuals, predictions),
            "mape": mean_absolute_percentage_error(actuals, predictions) * 100,
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
        })
    
    return pd.DataFrame(leaderboard).sort_values("mape").reset_index(drop=True)

def forecast_with_winner(monthly, models, leaderboard, horizon=FORECAST_HORIZON):
    """Refit the winner on full history and forecast forward."""
    winner_row = leaderboard.iloc[0]
    winner = next(model for model in models if model.name == winner_row["model"])
    
    winner.fit(monthly)
    predicted_values = winner.predict(horizon)
    
    future_dates = pd.date_range(start=monthly["ds"].max() + pd.DateOffset(months=1), 
                                 periods=horizon, freq="MS")
    margin = 1.96 * winner_row["rmse"]
    
    forecast = pd.DataFrame({
        "forecast_date": future_dates,
        "predicted_denial_rate": predicted_values,
        "lower_bound": predicted_values - margin,
        "upper_bound": predicted_values + margin,
    })
    
    # Clip to valid range
    for col in ["predicted_denial_rate", "lower_bound", "upper_bound"]:
        forecast[col] = forecast[col].clip(0, 100)
    
    forecast["model_used"] = winner.name
    return forecast

def save_forecast(forecast, table_name=OUTPUT_TABLE):
    try:
        spark.createDataFrame(forecast).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(table_name)
        print(f"Saved to {table_name}")
    except NameError:
        forecast.to_csv("denial_rate_forecast.csv", index=False)
        print("Saved to denial_rate_forecast.csv")

# Main execution
def main():
    monthly = load_monthly_denial_rate(CLAIMS_PATH)
    print(f"Working with {len(monthly)} months")
    
    models = [
        MovingAverage(),
        AutoArima(),
        SeasonalArima(),
        ProphetModel(),
        XgboostOnLagFeatures(LAGS),
    ]
    
    print("\nTesting models...")
    leaderboard = walk_forward_validate(monthly, models)
    print("\n" + leaderboard.to_string(index=False))
    
    winner = leaderboard.iloc[0]
    print(f"\n🏆 Best: {winner['model']} — {winner['mape']:.2f}% error")
    
    forecast = forecast_with_winner(monthly, models, leaderboard)
    print(f"\n{FORECAST_HORIZON}-month forecast:")
    print(forecast.to_string(index=False))
    
    save_forecast(forecast)

if __name__ == "__main__":
    main()
