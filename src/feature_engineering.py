from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_weekly_demand(path: Path | None = None) -> pd.DataFrame:
    weekly_path = path or PROCESSED_DIR / "weekly_demand.csv"
    return pd.read_csv(weekly_path, parse_dates=["date"])


def create_modeling_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    df = weekly_df.copy().sort_values(["store_id", "item_id", "week_id"]).reset_index(drop=True)
    demand_by_series = df.groupby(["store_id", "item_id"])["weekly_demand"]
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = demand_by_series.shift(lag)
    lagged = demand_by_series.shift(1)
    df["rolling_mean_4"] = lagged.rolling(4, min_periods=1).mean()
    df["rolling_mean_8"] = lagged.rolling(8, min_periods=1).mean()
    df["rolling_std_4"] = lagged.rolling(4, min_periods=1).std().fillna(0)
    return df


def main() -> None:
    weekly = load_weekly_demand()
    features = create_modeling_features(weekly)
    write_dataframe(features, PROCESSED_DIR / "weekly_demand_features.csv")
    LOGGER.info("Saved modeling feature table to %s", PROCESSED_DIR / "weekly_demand_features.csv")


if __name__ == "__main__":
    main()
