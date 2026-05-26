from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_inputs(
    forecast_path: Path | None = None,
    weekly_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_df = pd.read_csv(forecast_path or OUTPUT_DIR / "forecast_results.csv", parse_dates=["date"])
    weekly_df = pd.read_csv(weekly_path or PROCESSED_DIR / "weekly_demand.csv", parse_dates=["date"])
    return forecast_df, weekly_df


def bottom_up_reconciliation(forecast_df: pd.DataFrame) -> pd.DataFrame:
    detail = forecast_df.copy()
    detail["method"] = "bottom_up"
    category_rollup = (
        detail.groupby(["store_id", "cat_id", "week_id", "date"], as_index=False)["forecast"]
        .sum()
    )
    category_rollup["series_id"] = category_rollup["store_id"] + "|" + category_rollup["cat_id"]
    category_rollup["series_level"] = "store_category_reconciled"
    category_rollup["method"] = "bottom_up"
    return pd.concat([detail, category_rollup], ignore_index=True, sort=False)


def top_down_reconciliation(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    network = forecast_df.groupby(["week_id", "date"], as_index=False)["forecast"].sum()
    hist_props = (
        weekly_df.groupby(["store_id", "cat_id"], as_index=False)["weekly_demand"].sum()
    )
    hist_props["share"] = hist_props["weekly_demand"] / hist_props["weekly_demand"].sum()
    top_down = network.merge(hist_props[["store_id", "cat_id", "share"]], how="cross")
    top_down["forecast"] = top_down["forecast"] * top_down["share"]
    top_down["series_id"] = top_down["store_id"] + "|" + top_down["cat_id"]
    top_down["series_level"] = "store_category"
    top_down["method"] = "top_down"
    return top_down


def middle_out_reconciliation(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    category = (
        forecast_df.groupby(["store_id", "cat_id", "week_id", "date"], as_index=False)["forecast"]
        .sum()
    )
    item_props = (
        weekly_df.groupby(["store_id", "cat_id", "item_id"], as_index=False)["weekly_demand"].sum()
    )
    item_props["share"] = item_props["weekly_demand"] / item_props.groupby(["store_id", "cat_id"])["weekly_demand"].transform("sum")
    middle = category.merge(item_props[["store_id", "cat_id", "item_id", "share"]], on=["store_id", "cat_id"], how="left")
    middle["forecast"] = middle["forecast"] * middle["share"].fillna(0)
    middle["series_id"] = middle["store_id"] + "|" + middle["item_id"]
    middle["series_level"] = "store_item"
    middle["method"] = "middle_out"
    return middle


def simplified_mint(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    bu = bottom_up_reconciliation(forecast_df)
    td = top_down_reconciliation(forecast_df, weekly_df)
    detail = bu[bu["series_level"].astype(str).str.contains("store_category") == False].copy()
    category_bu = bu[bu["series_level"] == "store_category_reconciled"][["store_id", "cat_id", "week_id", "date", "forecast"]].rename(columns={"forecast": "bu_forecast"})
    category_td = td[["store_id", "cat_id", "week_id", "date", "forecast"]].rename(columns={"forecast": "td_forecast"})
    combined = category_bu.merge(category_td, on=["store_id", "cat_id", "week_id", "date"], how="inner")
    combined["forecast"] = 0.7 * combined["bu_forecast"] + 0.3 * combined["td_forecast"]
    combined["series_id"] = combined["store_id"] + "|" + combined["cat_id"]
    combined["series_level"] = "store_category"
    combined["method"] = "simplified_mint"
    return combined


def main() -> None:
    forecast_df, weekly_df = load_inputs()
    bu = bottom_up_reconciliation(forecast_df)
    td = top_down_reconciliation(forecast_df, weekly_df)
    mo = middle_out_reconciliation(forecast_df, weekly_df)
    mint_like = simplified_mint(forecast_df, weekly_df)

    reconciled = pd.concat([bu, td, mo, mint_like], ignore_index=True, sort=False)
    comparison = (
        reconciled.groupby(["method", "series_level"], as_index=False)["forecast"]
        .sum()
        .rename(columns={"forecast": "total_forecast"})
    )

    write_dataframe(reconciled, OUTPUT_DIR / "reconciled_forecast.csv")
    write_dataframe(comparison, OUTPUT_DIR / "reconciliation_comparison.csv")
    LOGGER.info("Saved reconciliation outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

