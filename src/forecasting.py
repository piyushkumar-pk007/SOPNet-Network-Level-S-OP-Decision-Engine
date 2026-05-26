from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import OUTPUT_DIR, PROCESSED_DIR, SAMPLING_CONFIG
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
        base = weekly_df.groupby(
            ["state_id", "store_id", "cat_id", "week_id", "date", "month", "quarter", "year", "week_of_year"],
            as_index=False,
        ).agg(
            weekly_demand=("weekly_demand", "sum"),
            avg_sell_price=("avg_sell_price", "mean"),
            is_event_week=("is_event_week", "max"),
        )
        base["series_id"] = base["store_id"] + "|" + base["cat_id"]
        base["series_level"] = "store_category"
        return base

    base = weekly_df.copy()
    base["series_id"] = base["store_id"] + "|" + base["item_id"]
    base["series_level"] = "store_item"
    return base


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


def _feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
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
    return [col for col in candidates if col in df.columns]


def _build_ml_model():
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    if LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
    return RandomForestRegressor(n_estimators=250, random_state=42)


def ml_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    supervised = create_supervised_frame(combined)
    train_supervised = create_supervised_frame(train_df)
    split_point = len(train_supervised)
    if split_point == 0 or supervised.empty:
        return moving_average(train_df["weekly_demand"], len(test_df))

    features = _feature_columns(supervised)
    train_part = supervised.iloc[:split_point].copy()
    test_part = supervised.iloc[split_point:].copy()
    if test_part.empty or train_part.empty:
        return moving_average(train_df["weekly_demand"], len(test_df))

    model = _build_ml_model()
    model.fit(train_part[features], train_part["weekly_demand"])
    preds = model.predict(test_part[features])
    if len(preds) < len(test_df):
        preds = np.concatenate([preds, np.repeat(preds[-1], len(test_df) - len(preds))])
    return preds[: len(test_df)]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def _model_registry() -> dict[str, Callable]:
    return {
        "seasonal_naive": lambda train, test: seasonal_naive(train["weekly_demand"], len(test)),
        "moving_average": lambda train, test: moving_average(train["weekly_demand"], len(test)),
        "sarima": lambda train, test: sarima_forecast(train["weekly_demand"], len(test)),
        "prophet_optional": prophet_forecast,
        "xgboost_or_lgbm_or_rf": ml_forecast,
    }


def _cross_validate_models(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    models = _model_registry()
    splitter = TimeSeriesSplit(n_splits=min(3, max(2, len(series) // horizon - 1)))
    rows = []
    series_id = series["series_id"].iloc[0]
    series_level = series["series_level"].iloc[0]
    state_id = series["state_id"].iloc[0]
    store_id = series["store_id"].iloc[0]

    for fold, (train_idx, test_idx) in enumerate(splitter.split(series), start=1):
        train = series.iloc[train_idx].copy()
        test = series.iloc[test_idx].copy()
        for model_name, model_fn in models.items():
            try:
                preds = np.asarray(model_fn(train, test), dtype=float)
                rows.append(
                    {
                        "series_id": series_id,
                        "series_level": series_level,
                        "state_id": state_id,
                        "store_id": store_id,
                        "target": "weekly_demand",
                        "model_name": model_name,
                        "fold": fold,
                        **calculate_metrics(test["weekly_demand"].to_numpy(dtype=float), preds),
                    }
                )
            except Exception as exc:
                LOGGER.warning("Model %s failed for %s: %s", model_name, series_id, exc)

    return pd.DataFrame(rows)


def _summarize_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    return (
        cv_results.groupby(["series_id", "series_level", "state_id", "store_id", "target", "model_name"], as_index=False)[
            ["mae", "rmse", "mape", "wape"]
        ]
        .mean()
        .sort_values(["series_id", "wape", "rmse"])
    )


def _best_model(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(["wape", "rmse"]).iloc[0]


def _forecast_rows(series: pd.DataFrame, test: pd.DataFrame, preds: np.ndarray, model_name: str) -> list[dict[str, object]]:
    rows = []
    for pred, (_, actual_row) in zip(preds, test.iterrows()):
        rows.append(
            {
                "series_id": series["series_id"].iloc[0],
                "series_level": series["series_level"].iloc[0],
                "state_id": series["state_id"].iloc[0],
                "store_id": series["store_id"].iloc[0],
                "cat_id": series["cat_id"].iloc[0] if "cat_id" in series.columns else "",
                "date": actual_row["date"],
                "week_id": actual_row["week_id"],
                "actual": actual_row["weekly_demand"],
                "forecast": float(pred),
                "model_name": model_name,
            }
        )
    return rows


def _best_model_forecast(series: pd.DataFrame, summary: pd.DataFrame, horizon: int) -> pd.DataFrame:
    models = _model_registry()
    train = series.iloc[:-horizon].copy()
    test = series.iloc[-horizon:].copy()
    best = _best_model(summary)
    preds = models[best["model_name"]](train, test)
    return pd.DataFrame(_forecast_rows(series, test, preds, best["model_name"]))


def evaluate_series(series_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = series_df.sort_values("date").reset_index(drop=True)
    if len(series) < 20:
        return pd.DataFrame(), pd.DataFrame()

    horizon = min(SAMPLING_CONFIG.forecast_horizon_weeks, max(2, len(series) // 5))
    cv_results = _cross_validate_models(series, horizon)
    if cv_results.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = _summarize_results(cv_results)
    return summary, _best_model_forecast(series, summary, horizon)


def plot_selected_series(forecasts: pd.DataFrame) -> None:
    if forecasts.empty:
        return

    series_ids = forecasts["series_id"].drop_duplicates().head(4).tolist()
    plot_df = forecasts[forecasts["series_id"].isin(series_ids)].copy()
    plt.figure(figsize=(12, 7))
    for series_id in series_ids:
        series = plot_df[plot_df["series_id"] == series_id]
        plt.plot(series["date"], series["actual"], linestyle="-", marker="o", label=f"{series_id} actual")
        plt.plot(series["date"], series["forecast"], linestyle="--", marker="o", label=f"{series_id} forecast")
    plt.title("Forecast Plot for Selected Series")
    plt.xlabel("Date")
    plt.ylabel("Weekly Demand")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_plot_selected_skus.png", dpi=160)
    plt.close()


def main() -> None:
    weekly = load_weekly_demand()
    base = build_forecast_base(weekly)
    summaries = []
    forecasts = []
    for _, series in base.groupby("series_id"):
        summary, forecast = evaluate_series(series)
        if not summary.empty:
            summaries.append(summary)
        if not forecast.empty:
            forecasts.append(forecast)

    if not summaries or not forecasts:
        raise ValueError("No forecast results generated. Check whether the weekly dataset has enough history.")

    model_comparison = pd.concat(summaries, ignore_index=True)
    forecast_results = pd.concat(forecasts, ignore_index=True)
    accuracy_by_level = (
        model_comparison.groupby(["series_level", "model_name"], as_index=False)[["mae", "rmse", "mape", "wape"]]
        .mean()
        .sort_values(["series_level", "wape"])
    )

    write_dataframe(forecast_results, OUTPUT_DIR / "forecast_results.csv")
    write_dataframe(model_comparison, OUTPUT_DIR / "model_comparison.csv")
    write_dataframe(accuracy_by_level, OUTPUT_DIR / "forecast_accuracy_by_level.csv")
    plot_selected_series(forecast_results)
    LOGGER.info("Saved forecasting outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
