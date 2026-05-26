from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_inputs(
    sales_path: Path | None = None,
    calendar_path: Path | None = None,
    prices_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_df = pd.read_csv(sales_path or PROCESSED_DIR / "sales_sample.csv", low_memory=False)
    calendar_df = pd.read_csv(calendar_path or PROCESSED_DIR / "calendar_clean.csv", parse_dates=["date"])
    prices_df = pd.read_csv(prices_path or PROCESSED_DIR / "prices_clean.csv", low_memory=False)
    return sales_df, calendar_df, prices_df


def melt_sales_to_long(sales_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [col for col in sales_df.columns if col.startswith("d_")]
    return sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="daily_demand",
    )


def _mode_or_default(values: pd.Series, default: str = "No Event") -> str:
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else default


def enrich_daily_sales(
    daily_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    df = daily_df.merge(calendar_df, on="d", how="left")
    df = df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["is_event_week"] = (df["event_name_1"].fillna("No Event") != "No Event").astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_id"] = df["wm_yr_wk"]
    return df


def _aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.groupby(
        [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "week_id",
            "year",
            "month",
            "quarter",
            "week_of_year",
        ],
        as_index=False,
    ).agg(
        date=("date", "max"),
        weekly_demand=("daily_demand", "sum"),
        avg_sell_price=("sell_price", "mean"),
        is_event_week=("is_event_week", "max"),
        event_name=("event_name_1", _mode_or_default),
        event_type=("event_type_1", _mode_or_default),
        snap_ca=("snap_CA", "max"),
        snap_tx=("snap_TX", "max"),
        snap_wi=("snap_WI", "max"),
    )
    weekly["avg_sell_price"] = weekly["avg_sell_price"].fillna(weekly["avg_sell_price"].median())
    weekly["revenue"] = weekly["weekly_demand"] * weekly["avg_sell_price"]
    return weekly.sort_values(["item_id", "store_id", "week_id"]).reset_index(drop=True)


def _add_rolling_demand(weekly_df: pd.DataFrame) -> pd.DataFrame:
    demand = weekly_df.groupby(["item_id", "store_id"])["weekly_demand"]
    weekly_df["rolling_4w_demand"] = demand.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    weekly_df["rolling_8w_demand"] = demand.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    return weekly_df


def build_weekly_demand(
    daily_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    daily = enrich_daily_sales(daily_df, calendar_df, prices_df)
    weekly = _aggregate_weekly(daily)
    return _add_rolling_demand(weekly)


def main() -> None:
    sales, calendar, prices = load_inputs()
    daily = melt_sales_to_long(sales)
    weekly = build_weekly_demand(daily, calendar, prices)
    write_dataframe(weekly, PROCESSED_DIR / "weekly_demand.csv")
    LOGGER.info("Saved weekly demand dataset to %s", PROCESSED_DIR / "weekly_demand.csv")


if __name__ == "__main__":
    main()
