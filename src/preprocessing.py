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
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [column for column in sales_df.columns if column.startswith("d_")]
    long_df = sales_df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="daily_demand",
    )
    return long_df


def build_weekly_demand(
    long_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = long_df.merge(calendar_df, on="d", how="left")
    merged = merged.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    merged["is_event_week"] = (merged["event_name_1"].fillna("No Event") != "No Event").astype(int)
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["quarter"] = merged["date"].dt.quarter
    merged["week_of_year"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["week_id"] = merged["wm_yr_wk"]

    weekly = (
        merged.groupby(
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
        )
        .agg(
            date=("date", "max"),
            weekly_demand=("daily_demand", "sum"),
            avg_sell_price=("sell_price", "mean"),
            is_event_week=("is_event_week", "max"),
            event_name=("event_name_1", lambda x: x.mode().iloc[0] if not x.mode().empty else "No Event"),
            event_type=("event_type_1", lambda x: x.mode().iloc[0] if not x.mode().empty else "No Event"),
            snap_ca=("snap_CA", "max"),
            snap_tx=("snap_TX", "max"),
            snap_wi=("snap_WI", "max"),
        )
    )
    weekly["avg_sell_price"] = weekly["avg_sell_price"].fillna(weekly["avg_sell_price"].median())
    weekly["revenue"] = weekly["weekly_demand"] * weekly["avg_sell_price"]
    weekly = weekly.sort_values(["item_id", "store_id", "week_id"]).reset_index(drop=True)
    weekly["rolling_4w_demand"] = (
        weekly.groupby(["item_id", "store_id"])["weekly_demand"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )
    weekly["rolling_8w_demand"] = (
        weekly.groupby(["item_id", "store_id"])["weekly_demand"].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    )
    return weekly


def main() -> None:
    sales_df, calendar_df, prices_df = load_inputs()
    long_df = melt_sales_to_long(sales_df)
    weekly = build_weekly_demand(long_df, calendar_df, prices_df)
    write_dataframe(weekly, PROCESSED_DIR / "weekly_demand.csv")
    LOGGER.info("Saved weekly demand dataset to %s", PROCESSED_DIR / "weekly_demand.csv")


if __name__ == "__main__":
    main()

