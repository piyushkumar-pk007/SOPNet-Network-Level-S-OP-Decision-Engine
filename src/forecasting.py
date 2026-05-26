from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import OUTPUT_DIR, PROCESSED_DIR, SAMPLING_CONFIG
from src.feature_engineering import create_modeling_features
from src.utils import get_logger, mape, wape, write_dataframe


LOGGER = get_logger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


def load_weekly_demand(path: Path | None = None) -> pd.DataFrame:
    weekly_path = path or PROCESSED_DIR / "weekly_demand.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly demand at {weekly_path}. Run preprocessing first.")
    return pd.read_csv(weekly_path, parse_dates=["date"])


def build_forecast_base(weekly_df: pd.DataFrame) -> pd.DataFrame:
    if SAMPLING_CONFIG.optimization_aggregation_level == "category_store_week":
        forecast_base = (
            weekly_df.groupby(
                ["state_id", "store_id", "cat_id", "week_id", "date", "month", "quarter", "year", "week_of_year"],
                as_index=False,
            )
            .agg(
                weekly_demand=("weekly_demand", "sum"),
                avg_sell_price=("avg_sell_price", "mean"),
                is_event_week=("is_event_week", "max"),
            )
        )
        forecast_base["series_id"] = forecast_base["store_id"] + "|" + forecast_base["cat_id"]
        forecast_base["series_level"] = "store_category"
    else:
        forecast_base = weekly_df.copy()
        forecast_base["series_id"] = forecast_base["store_id"] + "|" + forecast_base["item_id"]
        forecast_base["series_level"] = "store_item"
    return forecast_base


def seasonal_naive(train: pd.Series, horizon: int, seasonal_period: int = 4) -> np.ndarray:
    if len(train) < seasonal_period:
        return np.repeat(train.iloc[-1], horizon)
    return np.resize(train.iloc[-seasonal_period:].to_numpy(), horizon)


def moving_average(train: pd.Series, horizon: int, window: int = 4) -> np.ndarray:
    return np.repeat(train.iloc[-window:].mean() if len(train) >= window else train.mean(), horizon)


def sarima_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if SARIMAX is None or len(train) < 10:
        return moving_average(train, horizon)
    try:
        model = SARIMAX(
            train.astype(float),
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 4),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        return fitted.forecast(horizon).to_numpy()
    except Exception as exc:
        LOGGER.warning("SARIMA failed: %s", exc)
        return moving_average(train, horizon)


def prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    if Prophet is None:
        return moving_average(train_df["weekly_demand"], len(test_df))
    prophet_train = train_df[["date", "weekly_demand"]].rename(columns={"date": "ds", "weekly_demand": "y"})
    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=len(test_df), freq="W")
    forecast = model.predict(future)
    return forecast["yhat"].tail(len(test_df)).to_numpy()


def create_supervised_frame(series_df: pd.DataFrame) -> pd.DataFrame:
    df = series_df.copy().sort_values("date").reset_index(drop=True)
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df["weekly_demand"].shift(lag)
    df["rolling_mean_4"] = df["weekly_demand"].shift(1).rolling(4, min_periods=1).mean()
    df["rolling_mean_8"] = df["weekly_demand"].shift(1).rolling(8, min_periods=1).mean()
    df["rolling_std_4"] = df["weekly_demand"].shift(1).rolling(4, min_periods=1).std().fillna(0)
    return df.dropna().reset_index(drop=True)


def ml_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    supervised = create_supervised_frame(combined)
    train_supervised = create_supervised_frame(train_df)
    split_point = len(train_supervised)
    if split_point == 0 or supervised.empty:
        return moving_average(train_df["weekly_demand"], len(test_df))

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_std_4",
        "month",
        "quarter",
        "week_of_year",
        "is_event_week",
        "avg_sell_price",
    ]
    available = [column for column in feature_cols if column in supervised.columns]
    train_part = supervised.iloc[:split_point].copy()
    test_part = supervised.iloc[split_point:].copy()
    if test_part.empty or train_part.empty:
        return moving_average(train_df["weekly_demand"], len(test_df))

    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    elif LGBMRegressor is not None:
        model = LGBMRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=250, random_state=42)

    model.fit(train_part[available], train_part["weekly_demand"])
    preds = model.predict(test_part[available])
    if len(preds) < len(test_df):
        preds = np.concatenate([preds, np.repeat(preds[-1], len(test_df) - len(preds))])
    return preds[: len(test_df)]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def evaluate_series(series_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = series_df.sort_values("date").reset_index(drop=True)
    if len(data) < 20:
        return pd.DataFrame(), pd.DataFrame()
    horizon = min(SAMPLING_CONFIG.forecast_horizon_weeks, max(2, len(data) // 5))
    splitter = TimeSeriesSplit(n_splits=min(3, max(2, len(data) // horizon - 1)))
    rows = []
    model_fns: Dict[str, Callable] = {
        "seasonal_naive": lambda tr, te: seasonal_naive(tr["weekly_demand"], len(te)),
        "moving_average": lambda tr, te: moving_average(tr["weekly_demand"], len(te)),
        "sarima": lambda tr, te: sarima_forecast(tr["weekly_demand"], len(te)),
        "prophet_optional": prophet_forecast,
        "xgboost_or_lgbm_or_rf": ml_forecast,
    }

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data), start=1):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()
        for model_name, model_fn in model_fns.items():
            try:
                preds = np.asarray(model_fn(train_df, test_df), dtype=float)
                metrics = metric_dict(test_df["weekly_demand"].to_numpy(dtype=float), preds)
                rows.append(
                    {
                        "series_id": data["series_id"].iloc[0],
                        "series_level": data["series_level"].iloc[0],
                        "state_id": data["state_id"].iloc[0],
                        "store_id": data["store_id"].iloc[0],
                        "target": "weekly_demand",
                        "model_name": model_name,
                        "fold": fold,
                        **metrics,
                    }
                )
            except Exception as exc:
                LOGGER.warning("Model %s failed for %s: %s", model_name, data["series_id"].iloc[0], exc)

    results = pd.DataFrame(rows)
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        results.groupby(["series_id", "series_level", "state_id", "store_id", "target", "model_name"], as_index=False)[
            ["mae", "rmse", "mape", "wape"]
        ]
        .mean()
        .sort_values(["series_id", "wape", "rmse"])
    )

    best_model = summary.groupby("series_id", as_index=False).first()
    forecast_rows = []
    final_train = data.iloc[:-horizon].copy()
    final_test = data.iloc[-horizon:].copy()
    for _, row in best_model.iterrows():
        model_name = row["model_name"]
        preds = model_fns[model_name](final_train, final_test)
        for idx, (_, test_row) in enumerate(final_test.iterrows()):
            forecast_rows.append(
                {
                    "series_id": row["series_id"],
                    "series_level": row["series_level"],
                    "state_id": row["state_id"],
                    "store_id": row["store_id"],
                    "cat_id": data["cat_id"].iloc[0] if "cat_id" in data.columns else "",
                    "date": test_row["date"],
                    "week_id": test_row["week_id"],
                    "actual": test_row["weekly_demand"],
                    "forecast": float(preds[idx]),
                    "model_name": model_name,
                }
            )
    return summary, pd.DataFrame(forecast_rows)


def plot_selected_series(forecast_results: pd.DataFrame) -> None:
    if forecast_results.empty:
        return
    sample_series = forecast_results["series_id"].drop_duplicates().head(4).tolist()
    plot_df = forecast_results[forecast_results["series_id"].isin(sample_series)].copy()
    plt.figure(figsize=(12, 7))
    for series_id in sample_series:
        series_frame = plot_df[plot_df["series_id"] == series_id]
        plt.plot(series_frame["date"], series_frame["actual"], linestyle="-", marker="o", label=f"{series_id} actual")
        plt.plot(series_frame["date"], series_frame["forecast"], linestyle="--", marker="o", label=f"{series_id} forecast")
    plt.title("Forecast Plot for Selected Series")
    plt.xlabel("Date")
    plt.ylabel("Weekly Demand")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_plot_selected_skus.png", dpi=160)
    plt.close()


def main() -> None:
    weekly_df = load_weekly_demand()
    forecast_base = build_forecast_base(weekly_df)
    series_summaries = []
    series_forecasts = []
    for _, series_df in forecast_base.groupby("series_id"):
        summary, forecast = evaluate_series(series_df)
        if not summary.empty:
            series_summaries.append(summary)
        if not forecast.empty:
            series_forecasts.append(forecast)

    if not series_summaries or not series_forecasts:
        raise ValueError("No forecast results generated. Check whether the weekly dataset has enough history.")

    model_comparison = pd.concat(series_summaries, ignore_index=True)
    forecast_results = pd.concat(series_forecasts, ignore_index=True)
    best_by_level = (
        model_comparison.groupby(["series_level", "model_name"], as_index=False)[["mae", "rmse", "mape", "wape"]]
        .mean()
        .sort_values(["series_level", "wape"])
    )

    write_dataframe(forecast_results, OUTPUT_DIR / "forecast_results.csv")
    write_dataframe(model_comparison, OUTPUT_DIR / "model_comparison.csv")
    write_dataframe(best_by_level, OUTPUT_DIR / "forecast_accuracy_by_level.csv")
    plot_selected_series(forecast_results)
    LOGGER.info("Saved forecasting outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

