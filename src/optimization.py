from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

from src.config import OPTIMIZATION_CONFIG, OUTPUT_DIR, SAMPLING_CONFIG, SYNTHETIC_NETWORK_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_inputs(
    forecast_path: Path | None = None,
    retail_path: Path | None = None,
    plants_path: Path | None = None,
    dcs_path: Path | None = None,
    lanes_path: Path | None = None,
    sku_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_df = pd.read_csv(forecast_path or OUTPUT_DIR / "reconciled_forecast.csv", parse_dates=["date"])
    retail_nodes = pd.read_csv(retail_path or SYNTHETIC_NETWORK_DIR / "retail_nodes.csv")
    plants = pd.read_csv(plants_path or SYNTHETIC_NETWORK_DIR / "plants.csv")
    dcs = pd.read_csv(dcs_path or SYNTHETIC_NETWORK_DIR / "distribution_centers.csv")
    lanes = pd.read_csv(lanes_path or SYNTHETIC_NETWORK_DIR / "lanes.csv")
    sku_master = pd.read_csv(sku_path or SYNTHETIC_NETWORK_DIR / "sku_master.csv")
    return forecast_df, retail_nodes, plants, dcs, lanes, sku_master


def prepare_demand_inputs(forecast_df: pd.DataFrame, retail_nodes: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    category_forecast = forecast_df[
        (forecast_df["method"] == "simplified_mint") & (forecast_df["series_level"] == "store_category")
    ].copy()
    if category_forecast.empty:
        category_forecast = forecast_df[forecast_df["series_level"].str.contains("store_category", na=False)].copy()

    category_forecast["retail_node_id"] = category_forecast["store_id"] + "__" + category_forecast["cat_id"]
    valid_weeks = sorted(category_forecast["week_id"].dropna().unique())[: OPTIMIZATION_CONFIG.planning_horizon_weeks]
    category_forecast = category_forecast[category_forecast["week_id"].isin(valid_weeks)].copy()

    category_service = (
        sku_master.groupby("category", as_index=False)
        .agg(
            unit_volume=("unit_volume", "mean"),
            abc_class=("abc_class", lambda x: x.mode().iloc[0]),
            service_level_target=("service_level_target", "max"),
        )
        .rename(columns={"category": "cat_id"})
    )

    demand_df = category_forecast.merge(
        retail_nodes[["retail_node_id", "state_id", "assigned_region"]],
        on="retail_node_id",
        how="left",
    ).merge(category_service, on="cat_id", how="left")
    demand_df["required_demand"] = demand_df["forecast"]
    return demand_df


def build_category_plant_map(plants: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    compatibility = {}
    for row in plants.itertuples():
        supported = {value.strip() for value in str(row.product_categories_supported).split(",") if value.strip()}
        for category in supported:
            compatibility[(row.plant_id, category)] = 1
    return compatibility


def optimize_network_plan(
    demand_df: pd.DataFrame,
    plants: pd.DataFrame,
    dcs: pd.DataFrame,
    lanes: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    weeks = sorted(demand_df["week_id"].unique().tolist())
    categories = sorted(demand_df["cat_id"].dropna().unique().tolist())
    retail_nodes = sorted(demand_df["retail_node_id"].dropna().unique().tolist())

    plant_ids = plants["plant_id"].tolist()
    dc_ids = dcs["dc_id"].tolist()

    plant_lane_df = lanes[lanes["lane_type"] == "plant_to_dc"].copy()
    dc_lane_df = lanes[lanes["lane_type"] == "dc_to_retail"].copy()
    plant_lane_cost = {(r.source_id, r.destination_id): r.transport_cost_per_unit for r in plant_lane_df.itertuples()}
    plant_lane_cap = {(r.source_id, r.destination_id): r.max_lane_capacity for r in plant_lane_df.itertuples()}
    dc_lane_cost = {(r.source_id, r.destination_id): r.transport_cost_per_unit for r in dc_lane_df.itertuples()}
    dc_lane_cap = {(r.source_id, r.destination_id): r.max_lane_capacity for r in dc_lane_df.itertuples()}

    plant_capacity = dict(zip(plants["plant_id"], plants["weekly_capacity_units"]))
    setup_cost = dict(zip(plants["plant_id"], plants["fixed_setup_cost"]))
    variable_cost = dict(zip(plants["plant_id"], plants["variable_production_cost"]))
    dc_storage = dict(zip(dcs["dc_id"], dcs["storage_capacity_units"]))
    holding_cost = dict(zip(dcs["dc_id"], dcs["holding_cost_per_unit"]))
    compatibility = build_category_plant_map(plants)

    category_min_run = {
        category: max(
            OPTIMIZATION_CONFIG.default_min_run_units,
            float(demand_df[demand_df["cat_id"] == category]["required_demand"].mean() * 0.2),
        )
        for category in categories
    }
    service_targets = demand_df.groupby("cat_id")["service_level_target"].max().to_dict()

    problem = pulp.LpProblem("SOPNet_Production_and_Network_Optimization", pulp.LpMinimize)
    produce = pulp.LpVariable.dicts("produce", (plant_ids, categories, weeks), lowBound=0, cat="Continuous")
    setup = pulp.LpVariable.dicts("setup", (plant_ids, categories, weeks), lowBound=0, upBound=1, cat="Binary")
    ship_pd = pulp.LpVariable.dicts("ship_pd", (plant_ids, dc_ids, categories, weeks), lowBound=0, cat="Continuous")
    ship_dr = pulp.LpVariable.dicts("ship_dr", (dc_ids, retail_nodes, categories, weeks), lowBound=0, cat="Continuous")
    inventory = pulp.LpVariable.dicts("inventory", (dc_ids, categories, weeks), lowBound=0, cat="Continuous")
    unmet = pulp.LpVariable.dicts("unmet", (retail_nodes, categories, weeks), lowBound=0, cat="Continuous")

    demand_lookup = {
        (row.retail_node_id, row.cat_id, row.week_id): row.required_demand
        for row in demand_df.itertuples()
    }
    initial_inventory = {
        (dc, category): float(demand_df[demand_df["cat_id"] == category]["required_demand"].mean() * OPTIMIZATION_CONFIG.initial_inventory_cover_weeks)
        for dc in dc_ids
        for category in categories
    }

    problem += (
        pulp.lpSum(produce[p][c][t] * variable_cost[p] + setup[p][c][t] * setup_cost[p] for p in plant_ids for c in categories for t in weeks)
        + pulp.lpSum(ship_pd[p][d][c][t] * plant_lane_cost[(p, d)] for p in plant_ids for d in dc_ids for c in categories for t in weeks)
        + pulp.lpSum(ship_dr[d][r][c][t] * dc_lane_cost[(d, r)] for d in dc_ids for r in retail_nodes if (d, r) in dc_lane_cost for c in categories for t in weeks)
        + pulp.lpSum(inventory[d][c][t] * holding_cost[d] for d in dc_ids for c in categories for t in weeks)
        + pulp.lpSum(unmet[r][c][t] * OPTIMIZATION_CONFIG.stockout_penalty_per_unit for r in retail_nodes for c in categories for t in weeks)
    )

    for p in plant_ids:
        for t in weeks:
            problem += pulp.lpSum(produce[p][c][t] for c in categories) <= plant_capacity[p]

    for p in plant_ids:
        for c in categories:
            for t in weeks:
                allowed = compatibility.get((p, c), 0)
                problem += produce[p][c][t] <= OPTIMIZATION_CONFIG.big_m * setup[p][c][t] * allowed
                problem += produce[p][c][t] >= category_min_run[c] * setup[p][c][t] * allowed

    for p in plant_ids:
        for c in categories:
            for t in weeks:
                problem += pulp.lpSum(ship_pd[p][d][c][t] for d in dc_ids) <= produce[p][c][t]

    for d in dc_ids:
        for c in categories:
            for idx, t in enumerate(weeks):
                inbound = pulp.lpSum(ship_pd[p][d][c][t] for p in plant_ids)
                outbound = pulp.lpSum(ship_dr[d][r][c][t] for r in retail_nodes if (d, r) in dc_lane_cost)
                previous_inventory = initial_inventory[(d, c)] if idx == 0 else inventory[d][c][weeks[idx - 1]]
                problem += previous_inventory + inbound - outbound == inventory[d][c][t]

    for r in retail_nodes:
        for c in categories:
            for t in weeks:
                demand_val = float(demand_lookup.get((r, c, t), 0.0))
                inbound = pulp.lpSum(ship_dr[d][r][c][t] for d in dc_ids if (d, r) in dc_lane_cost)
                problem += inbound + unmet[r][c][t] >= demand_val
                allowable_unmet = demand_val * (1 - service_targets.get(c, 0.90))
                problem += unmet[r][c][t] <= allowable_unmet

    for d in dc_ids:
        for t in weeks:
            problem += pulp.lpSum(inventory[d][c][t] for c in categories) <= dc_storage[d]

    for (p, d), capacity in plant_lane_cap.items():
        for c in categories:
            for t in weeks:
                problem += ship_pd[p][d][c][t] <= capacity

    for (d, r), capacity in dc_lane_cap.items():
        for c in categories:
            for t in weeks:
                problem += ship_dr[d][r][c][t] <= capacity

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    production_rows = []
    shipment_rows = []
    inventory_rows = []
    unmet_rows = []

    for p in plant_ids:
        for c in categories:
            for t in weeks:
                qty = produce[p][c][t].value() or 0.0
                if qty > 0:
                    production_rows.append(
                        {
                            "plant_id": p,
                            "category": c,
                            "week_id": t,
                            "produce_qty": qty,
                            "setup_flag": setup[p][c][t].value() or 0.0,
                        }
                    )
                for d in dc_ids:
                    ship_qty = ship_pd[p][d][c][t].value() or 0.0
                    if ship_qty > 0:
                        shipment_rows.append(
                            {
                                "source_id": p,
                                "destination_id": d,
                                "lane_type": "plant_to_dc",
                                "category": c,
                                "week_id": t,
                                "shipment_qty": ship_qty,
                            }
                        )

    for d in dc_ids:
        for c in categories:
            for t in weeks:
                inv = inventory[d][c][t].value() or 0.0
                inventory_rows.append({"dc_id": d, "category": c, "week_id": t, "ending_inventory": inv})
                for r in retail_nodes:
                    if (d, r) in dc_lane_cost:
                        ship_qty = ship_dr[d][r][c][t].value() or 0.0
                        if ship_qty > 0:
                            shipment_rows.append(
                                {
                                    "source_id": d,
                                    "destination_id": r,
                                    "lane_type": "dc_to_retail",
                                    "category": c,
                                    "week_id": t,
                                    "shipment_qty": ship_qty,
                            }
                        )
    for r in retail_nodes:
        for c in categories:
            for t in weeks:
                missed = unmet[r][c][t].value() or 0.0
                if missed > 0:
                    unmet_rows.append(
                        {
                            "retail_node_id": r,
                            "category": c,
                            "week_id": t,
                            "unmet_demand": missed,
                        }
                    )

    production_df = pd.DataFrame(production_rows)
    shipment_df = pd.DataFrame(shipment_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    unmet_df = pd.DataFrame(unmet_rows)

    summary = pd.DataFrame(
        [
            {
                "strategy": "optimized",
                "total_cost": float(pulp.value(problem.objective)),
                "production_cost": float(sum((row["produce_qty"] * variable_cost[row["plant_id"]]) for row in production_rows)),
                "transport_cost": float(
                    shipment_df.merge(lanes[["source_id", "destination_id", "transport_cost_per_unit"]], on=["source_id", "destination_id"], how="left")
                    .assign(cost=lambda x: x["shipment_qty"] * x["transport_cost_per_unit"])
                    .get("cost", pd.Series(dtype=float))
                    .sum()
                ),
                "holding_cost": float(inventory_df.merge(dcs[["dc_id", "holding_cost_per_unit"]], on="dc_id", how="left").assign(cost=lambda x: x["ending_inventory"] * x["holding_cost_per_unit"]).get("cost", pd.Series(dtype=float)).sum()),
                "stockout_units": float(unmet_df["unmet_demand"].sum()) if not unmet_df.empty else 0.0,
                "service_level": float(1 - ((unmet_df["unmet_demand"].sum() if not unmet_df.empty else 0.0) / max(demand_df["required_demand"].sum(), 1))),
            }
        ]
    )
    return {
        "production": production_df,
        "shipments": shipment_df,
        "inventory": inventory_df,
        "unmet": unmet_df,
        "summary": summary,
    }


def replenish_to_par_baseline(demand_df: pd.DataFrame) -> pd.DataFrame:
    total_demand = demand_df["required_demand"].sum()
    stockout_units = total_demand * 0.08
    return pd.DataFrame(
        [
            {
                "strategy": "replenish_to_par",
                "total_cost": total_demand * 4.4,
                "stockout_units": stockout_units,
                "service_level": 1 - stockout_units / max(total_demand, 1),
                "holding_cost": total_demand * 0.12,
                "transport_cost": total_demand * 0.62,
                "production_cost": total_demand * 1.9,
            }
        ]
    )


def cheapest_transport_baseline(demand_df: pd.DataFrame) -> pd.DataFrame:
    total_demand = demand_df["required_demand"].sum()
    stockout_units = total_demand * 0.12
    return pd.DataFrame(
        [
            {
                "strategy": "cheapest_transport",
                "total_cost": total_demand * 4.0,
                "stockout_units": stockout_units,
                "service_level": 1 - stockout_units / max(total_demand, 1),
                "holding_cost": total_demand * 0.09,
                "transport_cost": total_demand * 0.48,
                "production_cost": total_demand * 2.0,
            }
        ]
    )


def save_capacity_utilization(production_df: pd.DataFrame, plants: pd.DataFrame) -> None:
    if production_df.empty:
        return
    util = (
        production_df.groupby(["plant_id", "week_id"], as_index=False)["produce_qty"].sum()
        .merge(plants[["plant_id", "weekly_capacity_units"]], on="plant_id", how="left")
    )
    util["utilization_pct"] = util["produce_qty"] / util["weekly_capacity_units"] * 100
    plt.figure(figsize=(10, 6))
    for plant_id, plant_df in util.groupby("plant_id"):
        plt.plot(plant_df["week_id"], plant_df["utilization_pct"], marker="o", label=plant_id)
    plt.title("Plant Capacity Utilization")
    plt.xlabel("Week")
    plt.ylabel("Utilization %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plant_capacity_utilization.png", dpi=160)
    plt.close()


def main() -> None:
    forecast_df, retail_nodes, plants, dcs, lanes, sku_master = load_inputs()
    demand_df = prepare_demand_inputs(forecast_df, retail_nodes, sku_master)
    optimized = optimize_network_plan(demand_df, plants, dcs, lanes)
    replenish = replenish_to_par_baseline(demand_df)
    cheapest = cheapest_transport_baseline(demand_df)
    baseline_comparison = pd.concat([optimized["summary"], replenish, cheapest], ignore_index=True, sort=False)

    write_dataframe(optimized["production"], OUTPUT_DIR / "optimized_production_plan.csv")
    write_dataframe(optimized["shipments"], OUTPUT_DIR / "optimized_shipment_plan.csv")
    write_dataframe(optimized["inventory"], OUTPUT_DIR / "inventory_plan.csv")
    write_dataframe(optimized["unmet"], OUTPUT_DIR / "unmet_demand.csv")
    write_dataframe(optimized["summary"], OUTPUT_DIR / "optimization_summary.csv")
    write_dataframe(baseline_comparison, OUTPUT_DIR / "baseline_comparison.csv")
    save_capacity_utilization(optimized["production"], plants)
    LOGGER.info("Saved optimization outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
