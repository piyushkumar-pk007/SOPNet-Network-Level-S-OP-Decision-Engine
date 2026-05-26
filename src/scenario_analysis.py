from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR, SCENARIO_CONFIG, SYNTHETIC_NETWORK_DIR
from src.optimization import load_inputs as load_opt_inputs
from src.optimization import optimize_network_plan, prepare_demand_inputs
from src.simulation import run_simulation
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def _apply_scenario(
    scenario_name: str,
    demand_df: pd.DataFrame,
    plants: pd.DataFrame,
    lanes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand_adj = demand_df.copy()
    plants_adj = plants.copy()
    lanes_adj = lanes.copy()

    if scenario_name == "High Demand":
        demand_adj["required_demand"] *= 1 + SCENARIO_CONFIG.demand_shock_pct
    elif scenario_name == "Lead Time Delay":
        lanes_adj["lead_time_days"] *= 1 + SCENARIO_CONFIG.lead_time_shock_pct
    elif scenario_name == "Transport Cost Inflation":
        lanes_adj["transport_cost_per_unit"] *= 1 + SCENARIO_CONFIG.transport_cost_shock_pct
    elif scenario_name == "Plant Capacity Shock":
        plants_adj.loc[plants_adj["plant_id"] == "Plant_A", "weekly_capacity_units"] *= 1 - SCENARIO_CONFIG.plant_capacity_shock_pct
    elif scenario_name == "Premium Service Scenario":
        demand_adj.loc[demand_adj["abc_class"] == "A", "service_level_target"] = 0.98

    return demand_adj, plants_adj, lanes_adj


def run_scenarios() -> pd.DataFrame:
    forecast_df, retail_nodes, plants, dcs, lanes, sku_master = load_opt_inputs()
    base_demand = prepare_demand_inputs(forecast_df, retail_nodes, sku_master)

    scenarios = [
        "Base Case",
        "High Demand",
        "Lead Time Delay",
        "Transport Cost Inflation",
        "Plant Capacity Shock",
        "Premium Service Scenario",
    ]
    rows = []
    for scenario_name in scenarios:
        demand_adj, plants_adj, lanes_adj = _apply_scenario(scenario_name, base_demand, plants, lanes)
        optimized = optimize_network_plan(demand_adj, plants_adj, dcs, lanes_adj)
        sim_base = demand_adj.copy()
        sim_base["forecast"] = sim_base["required_demand"]
        sim_base = sim_base.merge(
            optimized["shipments"][optimized["shipments"]["lane_type"] == "dc_to_retail"].groupby(["destination_id", "category", "week_id"], as_index=False)["shipment_qty"].sum().rename(columns={"destination_id": "retail_node_id", "category": "cat_id", "shipment_qty": "planned_supply"}),
            on=["retail_node_id", "cat_id", "week_id"],
            how="left",
        )
        sim_base["planned_supply"] = sim_base["planned_supply"].fillna(0)
        sim_results = run_simulation(sim_base)
        sim_summary = {
            "scenario": scenario_name,
            "total_cost": optimized["summary"]["total_cost"].iloc[0],
            "service_level": optimized["summary"]["service_level"].iloc[0],
            "stockout_units": optimized["summary"]["stockout_units"].iloc[0],
            "simulation_avg_service_level": sim_results["avg_service_level"].mean(),
            "simulation_avg_stockout_cost": sim_results["expected_stockout_cost"].mean(),
            "production_cost": optimized["summary"]["production_cost"].iloc[0],
            "transport_cost": optimized["summary"]["transport_cost"].iloc[0],
            "holding_cost": optimized["summary"]["holding_cost"].iloc[0],
        }
        rows.append(sim_summary)
    return pd.DataFrame(rows)


def save_scenario_plots(scenario_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(scenario_df["scenario"], scenario_df["simulation_avg_service_level"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average Service Level")
    plt.title("Scenario Service Level Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_service_level_chart.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(scenario_df["scenario"], scenario_df["total_cost"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Total Cost")
    plt.title("Scenario Cost Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_cost_chart.png", dpi=160)
    plt.close()


def main() -> None:
    scenario_df = run_scenarios()
    write_dataframe(scenario_df, OUTPUT_DIR / "scenario_comparison.csv")
    save_scenario_plots(scenario_df)
    LOGGER.info("Saved scenario outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
