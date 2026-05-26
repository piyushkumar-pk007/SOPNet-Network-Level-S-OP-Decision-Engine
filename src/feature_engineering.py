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
    group_cols = ["store_id", "item_id"]
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["weekly_demand"].shift(lag)
    df["rolling_mean_4"] = df.groupby(group_cols)["weekly_demand"].shift(1).rolling(4, min_periods=1).mean()
    df["rolling_mean_8"] = df.groupby(group_cols)["weekly_demand"].shift(1).rolling(8, min_periods=1).mean()
    df["rolling_std_4"] = df.groupby(group_cols)["weekly_demand"].shift(1).rolling(4, min_periods=1).std()
    df["rolling_std_4"] = df["rolling_std_4"].fillna(0)
    return df


def main() -> None:
    weekly_df = load_weekly_demand()
    feature_df = create_modeling_features(weekly_df)
    write_dataframe(feature_df, PROCESSED_DIR / "weekly_demand_features.csv")
    LOGGER.info("Saved modeling feature table to %s", PROCESSED_DIR / "weekly_demand_features.csv")


if __name__ == "__main__":
    main()

