from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SYNTHETIC_NETWORK_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_weekly_demand(path: Path | None = None) -> pd.DataFrame:
    weekly_path = path or PROCESSED_DIR / "weekly_demand.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly demand at {weekly_path}. Run preprocessing first.")
    return pd.read_csv(weekly_path, parse_dates=["date"])


def build_sku_master(weekly_df: pd.DataFrame) -> pd.DataFrame:
    sku_master = (
        weekly_df.groupby(["item_id", "cat_id", "dept_id"], as_index=False)
        .agg(total_demand=("weekly_demand", "sum"))
        .sort_values("total_demand", ascending=False)
        .reset_index(drop=True)
    )
    sku_master["demand_share"] = sku_master["total_demand"] / sku_master["total_demand"].sum()
    sku_master["cumulative_share"] = sku_master["demand_share"].cumsum()
    sku_master["abc_class"] = np.select(
        [sku_master["cumulative_share"] <= 0.70, sku_master["cumulative_share"] <= 0.90],
        ["A", "B"],
        default="C",
    )
    service_map = {"A": 0.95, "B": 0.90, "C": 0.85}
    sku_master["service_level_target"] = sku_master["abc_class"].map(service_map)
    sku_master["unit_volume"] = 0.05 + (sku_master.index % 5) * 0.01
    sku_master["unit_weight"] = 0.25 + (sku_master.index % 7) * 0.05
    sku_master = sku_master.rename(columns={"cat_id": "category", "dept_id": "department"})
    return sku_master


def build_plants(categories: list[str]) -> pd.DataFrame:
    rows = [
        {
            "plant_id": "Plant_A",
            "plant_name": "Plant_A",
            "weekly_capacity_units": 220_000,
            "fixed_setup_cost": 7_500,
            "variable_production_cost": 1.45,
            "product_categories_supported": ",".join(categories[: max(2, len(categories) // 2)]),
        },
        {
            "plant_id": "Plant_B",
            "plant_name": "Plant_B",
            "weekly_capacity_units": 180_000,
            "fixed_setup_cost": 6_500,
            "variable_production_cost": 1.60,
            "product_categories_supported": ",".join(categories[1:] if len(categories) > 1 else categories),
        },
        {
            "plant_id": "Plant_C",
            "plant_name": "Plant_C",
            "weekly_capacity_units": 200_000,
            "fixed_setup_cost": 7_000,
            "variable_production_cost": 1.55,
            "product_categories_supported": ",".join(categories),
        },
    ]
    return pd.DataFrame(rows)


def build_distribution_centers() -> pd.DataFrame:
    rows = [
        {"dc_id": "DC_West", "dc_name": "DC_West", "storage_capacity_units": 150_000, "holding_cost_per_unit": 0.08},
        {"dc_id": "DC_Central", "dc_name": "DC_Central", "storage_capacity_units": 140_000, "holding_cost_per_unit": 0.07},
        {"dc_id": "DC_South", "dc_name": "DC_South", "storage_capacity_units": 135_000, "holding_cost_per_unit": 0.075},
        {"dc_id": "DC_North", "dc_name": "DC_North", "storage_capacity_units": 145_000, "holding_cost_per_unit": 0.07},
        {"dc_id": "DC_East", "dc_name": "DC_East", "storage_capacity_units": 150_000, "holding_cost_per_unit": 0.08},
    ]
    return pd.DataFrame(rows)


def build_retail_nodes(weekly_df: pd.DataFrame) -> pd.DataFrame:
    nodes = (
        weekly_df[["store_id", "state_id", "cat_id"]]
        .drop_duplicates()
        .rename(columns={"cat_id": "category"})
        .copy()
    )
    state_region_map = {"CA": "West", "TX": "South", "WI": "North"}
    nodes["retail_node_id"] = nodes["store_id"] + "__" + nodes["category"]
    nodes["assigned_region"] = nodes["state_id"].str[:2].map(state_region_map).fillna("Central")
    return nodes[["retail_node_id", "store_id", "state_id", "category", "assigned_region"]]


def build_lanes(plants: pd.DataFrame, dcs: pd.DataFrame, retail_nodes: pd.DataFrame) -> pd.DataFrame:
    lane_rows = []
    plant_dc_costs = {
        "DC_West": 0.55,
        "DC_Central": 0.48,
        "DC_South": 0.52,
        "DC_North": 0.50,
        "DC_East": 0.56,
    }
    for plant_id in plants["plant_id"]:
        for dc_id in dcs["dc_id"]:
            lane_rows.append(
                {
                    "source_id": plant_id,
                    "destination_id": dc_id,
                    "lane_type": "plant_to_dc",
                    "transport_cost_per_unit": plant_dc_costs[dc_id] + (0.03 if plant_id == "Plant_B" else 0.0),
                    "lead_time_days": 4 if dc_id in {"DC_Central", "DC_North"} else 5,
                    "max_lane_capacity": 120_000,
                }
            )

    region_dc_map = {
        "West": "DC_West",
        "South": "DC_South",
        "North": "DC_North",
        "Central": "DC_Central",
        "East": "DC_East",
    }
    all_dcs = dcs["dc_id"].tolist()
    for row in retail_nodes.itertuples():
        preferred_dc = region_dc_map.get(row.assigned_region, "DC_Central")
        for dc_id in all_dcs:
            is_preferred = dc_id == preferred_dc
            lane_rows.append(
                {
                    "source_id": dc_id,
                    "destination_id": row.retail_node_id,
                    "lane_type": "dc_to_retail",
                    "transport_cost_per_unit": 0.30 if is_preferred else 0.42,
                    "lead_time_days": 2 if is_preferred else 4,
                    "max_lane_capacity": 40_000,
                }
            )
    return pd.DataFrame(lane_rows)


def main() -> None:
    weekly = load_weekly_demand()
    sku_master = build_sku_master(weekly)
    categories = sorted(sku_master["category"].unique().tolist())
    plants = build_plants(categories)
    dcs = build_distribution_centers()
    retail_nodes = build_retail_nodes(weekly)
    lanes = build_lanes(plants, dcs, retail_nodes)

    write_dataframe(plants, SYNTHETIC_NETWORK_DIR / "plants.csv")
    write_dataframe(dcs, SYNTHETIC_NETWORK_DIR / "distribution_centers.csv")
    write_dataframe(retail_nodes, SYNTHETIC_NETWORK_DIR / "retail_nodes.csv")
    write_dataframe(lanes, SYNTHETIC_NETWORK_DIR / "lanes.csv")
    write_dataframe(sku_master, SYNTHETIC_NETWORK_DIR / "sku_master.csv")
    LOGGER.info("Saved synthetic network tables to %s", SYNTHETIC_NETWORK_DIR)


if __name__ == "__main__":
    main()
