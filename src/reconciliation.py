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
    category = detail.groupby(["store_id", "cat_id", "week_id", "date"], as_index=False)["forecast"].sum()
    category["series_id"] = category["store_id"] + "|" + category["cat_id"]
    category["series_level"] = "store_category_reconciled"
    category["method"] = "bottom_up"
    return pd.concat([detail, category], ignore_index=True, sort=False)


def top_down_reconciliation(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    network = forecast_df.groupby(["week_id", "date"], as_index=False)["forecast"].sum()
    shares = weekly_df.groupby(["store_id", "cat_id"], as_index=False)["weekly_demand"].sum()
    shares["share"] = shares["weekly_demand"] / shares["weekly_demand"].sum()
    reconciled = network.merge(shares[["store_id", "cat_id", "share"]], how="cross")
    reconciled["forecast"] = reconciled["forecast"] * reconciled["share"]
    reconciled["series_id"] = reconciled["store_id"] + "|" + reconciled["cat_id"]
    reconciled["series_level"] = "store_category"
    reconciled["method"] = "top_down"
    return reconciled


def middle_out_reconciliation(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    category = (
        forecast_df.groupby(["store_id", "cat_id", "week_id", "date"], as_index=False)["forecast"]
        .sum()
    )
    item_shares = weekly_df.groupby(["store_id", "cat_id", "item_id"], as_index=False)["weekly_demand"].sum()
    item_shares["share"] = item_shares["weekly_demand"] / item_shares.groupby(["store_id", "cat_id"])["weekly_demand"].transform("sum")
    reconciled = category.merge(item_shares[["store_id", "cat_id", "item_id", "share"]], on=["store_id", "cat_id"], how="left")
    reconciled["forecast"] = reconciled["forecast"] * reconciled["share"].fillna(0)
    reconciled["series_id"] = reconciled["store_id"] + "|" + reconciled["item_id"]
    reconciled["series_level"] = "store_item"
    reconciled["method"] = "middle_out"
    return reconciled


def simplified_mint(forecast_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    bottom_up_df = bottom_up_reconciliation(forecast_df)
    top_down_df = top_down_reconciliation(forecast_df, weekly_df)
    category_bottom_up = bottom_up_df[bottom_up_df["series_level"] == "store_category_reconciled"][
        ["store_id", "cat_id", "week_id", "date", "forecast"]
    ].rename(columns={"forecast": "bu_forecast"})
    category_top_down = top_down_df[["store_id", "cat_id", "week_id", "date", "forecast"]].rename(columns={"forecast": "td_forecast"})
    combined = category_bottom_up.merge(category_top_down, on=["store_id", "cat_id", "week_id", "date"], how="inner")
    combined["forecast"] = 0.7 * combined["bu_forecast"] + 0.3 * combined["td_forecast"]
    combined["series_id"] = combined["store_id"] + "|" + combined["cat_id"]
    combined["series_level"] = "store_category"
    combined["method"] = "simplified_mint"
    return combined


def main() -> None:
    forecast, weekly = load_inputs()
    bottom_up = bottom_up_reconciliation(forecast)
    top_down = top_down_reconciliation(forecast, weekly)
    middle_out = middle_out_reconciliation(forecast, weekly)
    mint_like = simplified_mint(forecast, weekly)

    reconciled = pd.concat([bottom_up, top_down, middle_out, mint_like], ignore_index=True, sort=False)
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
