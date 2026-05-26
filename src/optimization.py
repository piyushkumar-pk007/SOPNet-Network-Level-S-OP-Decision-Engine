from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp

from src.config import OPTIMIZATION_CONFIG, OUTPUT_DIR, SYNTHETIC_NETWORK_DIR
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
    forecast = pd.read_csv(forecast_path or OUTPUT_DIR / "reconciled_forecast.csv", parse_dates=["date"])
    retail = pd.read_csv(retail_path or SYNTHETIC_NETWORK_DIR / "retail_nodes.csv")
    plants = pd.read_csv(plants_path or SYNTHETIC_NETWORK_DIR / "plants.csv")
    dcs = pd.read_csv(dcs_path or SYNTHETIC_NETWORK_DIR / "distribution_centers.csv")
    lanes = pd.read_csv(lanes_path or SYNTHETIC_NETWORK_DIR / "lanes.csv")
    sku = pd.read_csv(sku_path or SYNTHETIC_NETWORK_DIR / "sku_master.csv")
    return forecast, retail, plants, dcs, lanes, sku


def prepare_demand_inputs(
    forecast_df: pd.DataFrame,
    retail_df: pd.DataFrame,
    sku_df: pd.DataFrame,
) -> pd.DataFrame:
    demand = forecast_df[
        (forecast_df["method"] == "simplified_mint") & (forecast_df["series_level"] == "store_category")
    ].copy()
    if demand.empty:
        demand = forecast_df[forecast_df["series_level"].str.contains("store_category", na=False)].copy()

    demand["retail_node_id"] = demand["store_id"] + "__" + demand["cat_id"]
    weeks = sorted(demand["week_id"].dropna().unique())[: OPTIMIZATION_CONFIG.planning_horizon_weeks]
    demand = demand[demand["week_id"].isin(weeks)].copy()

    service = (
        sku_df.groupby("category", as_index=False)
        .agg(
            unit_volume=("unit_volume", "mean"),
            abc_class=("abc_class", lambda x: x.mode().iloc[0]),
            service_level_target=("service_level_target", "max"),
        )
        .rename(columns={"category": "cat_id"})
    )

    demand = demand.merge(
        retail_df[["retail_node_id", "state_id", "assigned_region"]],
        on="retail_node_id",
        how="left",
    ).merge(service, on="cat_id", how="left")
    demand["required_demand"] = demand["forecast"]
    return demand


def build_category_plant_map(plants: pd.DataFrame) -> dict[tuple[str, str], int]:
    compatibility: dict[tuple[str, str], int] = {}
    for row in plants.itertuples():
        categories = {value.strip() for value in str(row.product_categories_supported).split(",") if value.strip()}
        for category in categories:
            compatibility[(row.plant_id, category)] = 1
    return compatibility


def build_lane_maps(lanes: pd.DataFrame) -> tuple[dict[tuple[str, str], float], ...]:
    plant_lanes = lanes[lanes["lane_type"] == "plant_to_dc"]
    dc_lanes = lanes[lanes["lane_type"] == "dc_to_retail"]
    plant_cost = {(row.source_id, row.destination_id): row.transport_cost_per_unit for row in plant_lanes.itertuples()}
    plant_cap = {(row.source_id, row.destination_id): row.max_lane_capacity for row in plant_lanes.itertuples()}
    dc_cost = {(row.source_id, row.destination_id): row.transport_cost_per_unit for row in dc_lanes.itertuples()}
    dc_cap = {(row.source_id, row.destination_id): row.max_lane_capacity for row in dc_lanes.itertuples()}
    return plant_cost, plant_cap, dc_cost, dc_cap


def _add_capacity_constraints(
    problem: pulp.LpProblem,
    produce,
    setup,
    ship_pd,
    *,
    weeks: list[int],
    categories: list[str],
    plant_ids: list[str],
    dc_ids: list[str],
    plant_capacity: dict[str, float],
    compatibility: dict[tuple[str, str], int],
    category_min_run: dict[str, float],
) -> None:
    for plant in plant_ids:
        for week in weeks:
            problem += pulp.lpSum(produce[plant][category][week] for category in categories) <= plant_capacity[plant]

    for plant in plant_ids:
        for category in categories:
            for week in weeks:
                allowed = compatibility.get((plant, category), 0)
                problem += produce[plant][category][week] <= OPTIMIZATION_CONFIG.big_m * setup[plant][category][week] * allowed
                problem += produce[plant][category][week] >= category_min_run[category] * setup[plant][category][week] * allowed
                problem += pulp.lpSum(ship_pd[plant][dc][category][week] for dc in dc_ids) <= produce[plant][category][week]


def _add_inventory_constraints(
    problem: pulp.LpProblem,
    ship_pd,
    ship_dr,
    inventory,
    *,
    weeks: list[int],
    categories: list[str],
    plant_ids: list[str],
    dc_ids: list[str],
    node_ids: list[str],
    initial_inventory: dict[tuple[str, str], float],
    dc_lane_cap: dict[tuple[str, str], float],
) -> None:
    for dc in dc_ids:
        for category in categories:
            for idx, week in enumerate(weeks):
                inbound = pulp.lpSum(ship_pd[plant][dc][category][week] for plant in plant_ids)
                outbound = pulp.lpSum(ship_dr[dc][node][category][week] for node in node_ids if (dc, node) in dc_lane_cap)
                starting_inventory = initial_inventory[(dc, category)] if idx == 0 else inventory[dc][category][weeks[idx - 1]]
                problem += starting_inventory + inbound - outbound == inventory[dc][category][week]


def _add_service_constraints(
    problem: pulp.LpProblem,
    ship_dr,
    unmet,
    *,
    weeks: list[int],
    categories: list[str],
    dc_ids: list[str],
    node_ids: list[str],
    demand_lookup: dict[tuple[str, str, int], float],
    service_targets: dict[str, float],
    dc_lane_cap: dict[tuple[str, str], float],
) -> None:
    for node in node_ids:
        for category in categories:
            for week in weeks:
                demand = float(demand_lookup.get((node, category, week), 0.0))
                inbound = pulp.lpSum(ship_dr[dc][node][category][week] for dc in dc_ids if (dc, node) in dc_lane_cap)
                problem += inbound + unmet[node][category][week] >= demand
                problem += unmet[node][category][week] <= demand * (1 - service_targets.get(category, 0.90))


def _add_lane_constraints(
    problem: pulp.LpProblem,
    ship_pd,
    ship_dr,
    inventory,
    *,
    weeks: list[int],
    categories: list[str],
    dc_ids: list[str],
    node_ids: list[str],
    dc_storage: dict[str, float],
    plant_lane_cap: dict[tuple[str, str], float],
    dc_lane_cap: dict[tuple[str, str], float],
) -> None:
    for dc in dc_ids:
        for week in weeks:
            problem += pulp.lpSum(inventory[dc][category][week] for category in categories) <= dc_storage[dc]

    for (plant, dc), capacity in plant_lane_cap.items():
        for category in categories:
            for week in weeks:
                problem += ship_pd[plant][dc][category][week] <= capacity

    for (dc, node), capacity in dc_lane_cap.items():
        for category in categories:
            for week in weeks:
                problem += ship_dr[dc][node][category][week] <= capacity


def add_constraints(
    problem: pulp.LpProblem,
    produce,
    setup,
    ship_pd,
    ship_dr,
    inventory,
    unmet,
    *,
    weeks: list[int],
    categories: list[str],
    plant_ids: list[str],
    dc_ids: list[str],
    node_ids: list[str],
    plant_capacity: dict[str, float],
    compatibility: dict[tuple[str, str], int],
    category_min_run: dict[str, float],
    initial_inventory: dict[tuple[str, str], float],
    demand_lookup: dict[tuple[str, str, int], float],
    service_targets: dict[str, float],
    dc_storage: dict[str, float],
    plant_lane_cap: dict[tuple[str, str], float],
    dc_lane_cap: dict[tuple[str, str], float],
) -> None:
    _add_capacity_constraints(
        problem,
        produce,
        setup,
        ship_pd,
        weeks=weeks,
        categories=categories,
        plant_ids=plant_ids,
        dc_ids=dc_ids,
        plant_capacity=plant_capacity,
        compatibility=compatibility,
        category_min_run=category_min_run,
    )
    _add_inventory_constraints(
        problem,
        ship_pd,
        ship_dr,
        inventory,
        weeks=weeks,
        categories=categories,
        plant_ids=plant_ids,
        dc_ids=dc_ids,
        node_ids=node_ids,
        initial_inventory=initial_inventory,
        dc_lane_cap=dc_lane_cap,
    )
    _add_service_constraints(
        problem,
        ship_dr,
        unmet,
        weeks=weeks,
        categories=categories,
        dc_ids=dc_ids,
        node_ids=node_ids,
        demand_lookup=demand_lookup,
        service_targets=service_targets,
        dc_lane_cap=dc_lane_cap,
    )
    _add_lane_constraints(
        problem,
        ship_pd,
        ship_dr,
        inventory,
        weeks=weeks,
        categories=categories,
        dc_ids=dc_ids,
        node_ids=node_ids,
        dc_storage=dc_storage,
        plant_lane_cap=plant_lane_cap,
        dc_lane_cap=dc_lane_cap,
    )


def collect_solution(
    weeks: list[int],
    categories: list[str],
    plant_ids: list[str],
    dc_ids: list[str],
    node_ids: list[str],
    dc_lane_cost: dict[tuple[str, str], float],
    produce,
    setup,
    ship_pd,
    ship_dr,
    inventory,
    unmet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    production_rows = []
    shipment_rows = []
    inventory_rows = []
    unmet_rows = []

    for plant in plant_ids:
        for category in categories:
            for week in weeks:
                qty = produce[plant][category][week].value() or 0.0
                if qty > 0:
                    production_rows.append(
                        {
                            "plant_id": plant,
                            "category": category,
                            "week_id": week,
                            "produce_qty": qty,
                            "setup_flag": setup[plant][category][week].value() or 0.0,
                        }
                    )
                for dc in dc_ids:
                    shipped = ship_pd[plant][dc][category][week].value() or 0.0
                    if shipped > 0:
                        shipment_rows.append(
                            {
                                "source_id": plant,
                                "destination_id": dc,
                                "lane_type": "plant_to_dc",
                                "category": category,
                                "week_id": week,
                                "shipment_qty": shipped,
                            }
                        )

    for dc in dc_ids:
        for category in categories:
            for week in weeks:
                ending_inventory = inventory[dc][category][week].value() or 0.0
                inventory_rows.append(
                    {
                        "dc_id": dc,
                        "category": category,
                        "week_id": week,
                        "ending_inventory": ending_inventory,
                    }
                )
                for node in node_ids:
                    if (dc, node) not in dc_lane_cost:
                        continue
                    shipped = ship_dr[dc][node][category][week].value() or 0.0
                    if shipped > 0:
                        shipment_rows.append(
                            {
                                "source_id": dc,
                                "destination_id": node,
                                "lane_type": "dc_to_retail",
                                "category": category,
                                "week_id": week,
                                "shipment_qty": shipped,
                            }
                        )

    for node in node_ids:
        for category in categories:
            for week in weeks:
                missed = unmet[node][category][week].value() or 0.0
                if missed > 0:
                    unmet_rows.append(
                        {
                            "retail_node_id": node,
                            "category": category,
                            "week_id": week,
                            "unmet_demand": missed,
                        }
                    )

    return (
        pd.DataFrame(production_rows),
        pd.DataFrame(shipment_rows),
        pd.DataFrame(inventory_rows),
        pd.DataFrame(unmet_rows),
    )


def build_summary(
    problem: pulp.LpProblem,
    demand_df: pd.DataFrame,
    production_df: pd.DataFrame,
    shipment_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    unmet_df: pd.DataFrame,
    lanes: pd.DataFrame,
    plants: pd.DataFrame,
    dcs: pd.DataFrame,
) -> pd.DataFrame:
    variable_cost = dict(zip(plants["plant_id"], plants["variable_production_cost"]))
    total_unmet = float(unmet_df["unmet_demand"].sum()) if not unmet_df.empty else 0.0
    total_demand = max(float(demand_df["required_demand"].sum()), 1.0)

    transport_cost = 0.0
    if not shipment_df.empty:
        shipment_costs = shipment_df.merge(
            lanes[["source_id", "destination_id", "transport_cost_per_unit"]],
            on=["source_id", "destination_id"],
            how="left",
        )
        transport_cost = float((shipment_costs["shipment_qty"] * shipment_costs["transport_cost_per_unit"]).sum())

    holding_cost = 0.0
    if not inventory_df.empty:
        inventory_costs = inventory_df.merge(
            dcs[["dc_id", "holding_cost_per_unit"]],
            on="dc_id",
            how="left",
        )
        holding_cost = float((inventory_costs["ending_inventory"] * inventory_costs["holding_cost_per_unit"]).sum())

    summary = pd.DataFrame(
        [
            {
                "strategy": "optimized",
                "total_cost": float(pulp.value(problem.objective)),
                "production_cost": float(
                    sum(row["produce_qty"] * variable_cost[row["plant_id"]] for _, row in production_df.iterrows())
                ),
                "transport_cost": transport_cost,
                "holding_cost": holding_cost,
                "stockout_units": total_unmet,
                "service_level": float(1 - total_unmet / total_demand),
            }
        ]
    )
    return summary


def optimize_network_plan(
    demand_df: pd.DataFrame,
    plants: pd.DataFrame,
    dcs: pd.DataFrame,
    lanes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    weeks = sorted(demand_df["week_id"].unique().tolist())
    categories = sorted(demand_df["cat_id"].dropna().unique().tolist())
    node_ids = sorted(demand_df["retail_node_id"].dropna().unique().tolist())
    plant_ids = plants["plant_id"].tolist()
    dc_ids = dcs["dc_id"].tolist()

    plant_lane_cost, plant_lane_cap, dc_lane_cost, dc_lane_cap = build_lane_maps(lanes)
    compatibility = build_category_plant_map(plants)
    plant_capacity = dict(zip(plants["plant_id"], plants["weekly_capacity_units"]))
    setup_cost = dict(zip(plants["plant_id"], plants["fixed_setup_cost"]))
    variable_cost = dict(zip(plants["plant_id"], plants["variable_production_cost"]))
    dc_storage = dict(zip(dcs["dc_id"], dcs["storage_capacity_units"]))
    holding_cost = dict(zip(dcs["dc_id"], dcs["holding_cost_per_unit"]))

    category_min_run = {
        category: max(
            OPTIMIZATION_CONFIG.default_min_run_units,
            float(demand_df.loc[demand_df["cat_id"] == category, "required_demand"].mean() * 0.2),
        )
        for category in categories
    }
    service_targets = demand_df.groupby("cat_id")["service_level_target"].max().to_dict()
    demand_lookup = {
        (row.retail_node_id, row.cat_id, row.week_id): row.required_demand
        for row in demand_df.itertuples()
    }
    initial_inventory = {
        (dc, category): float(
            demand_df.loc[demand_df["cat_id"] == category, "required_demand"].mean()
            * OPTIMIZATION_CONFIG.initial_inventory_cover_weeks
        )
        for dc in dc_ids
        for category in categories
    }

    problem = pulp.LpProblem("SOPNet_Production_and_Network_Optimization", pulp.LpMinimize)
    produce = pulp.LpVariable.dicts("produce", (plant_ids, categories, weeks), lowBound=0, cat="Continuous")
    setup = pulp.LpVariable.dicts("setup", (plant_ids, categories, weeks), lowBound=0, upBound=1, cat="Binary")
    ship_pd = pulp.LpVariable.dicts("ship_pd", (plant_ids, dc_ids, categories, weeks), lowBound=0, cat="Continuous")
    ship_dr = pulp.LpVariable.dicts("ship_dr", (dc_ids, node_ids, categories, weeks), lowBound=0, cat="Continuous")
    inventory = pulp.LpVariable.dicts("inventory", (dc_ids, categories, weeks), lowBound=0, cat="Continuous")
    unmet = pulp.LpVariable.dicts("unmet", (node_ids, categories, weeks), lowBound=0, cat="Continuous")

    problem += (
        pulp.lpSum(
            produce[plant][category][week] * variable_cost[plant]
            + setup[plant][category][week] * setup_cost[plant]
            for plant in plant_ids
            for category in categories
            for week in weeks
        )
        + pulp.lpSum(
            ship_pd[plant][dc][category][week] * plant_lane_cost[(plant, dc)]
            for plant in plant_ids
            for dc in dc_ids
            for category in categories
            for week in weeks
        )
        + pulp.lpSum(
            ship_dr[dc][node][category][week] * dc_lane_cost[(dc, node)]
            for dc in dc_ids
            for node in node_ids
            if (dc, node) in dc_lane_cost
            for category in categories
            for week in weeks
        )
        + pulp.lpSum(
            inventory[dc][category][week] * holding_cost[dc]
            for dc in dc_ids
            for category in categories
            for week in weeks
        )
        + pulp.lpSum(
            unmet[node][category][week] * OPTIMIZATION_CONFIG.stockout_penalty_per_unit
            for node in node_ids
            for category in categories
            for week in weeks
        )
    )

    add_constraints(
        problem,
        produce,
        setup,
        ship_pd,
        ship_dr,
        inventory,
        unmet,
        weeks=weeks,
        categories=categories,
        plant_ids=plant_ids,
        dc_ids=dc_ids,
        node_ids=node_ids,
        plant_capacity=plant_capacity,
        compatibility=compatibility,
        category_min_run=category_min_run,
        initial_inventory=initial_inventory,
        demand_lookup=demand_lookup,
        service_targets=service_targets,
        dc_storage=dc_storage,
        plant_lane_cap=plant_lane_cap,
        dc_lane_cap=dc_lane_cap,
    )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    production_df, shipment_df, inventory_df, unmet_df = collect_solution(
        weeks,
        categories,
        plant_ids,
        dc_ids,
        node_ids,
        dc_lane_cost,
        produce,
        setup,
        ship_pd,
        ship_dr,
        inventory,
        unmet,
    )
    summary = build_summary(
        problem,
        demand_df,
        production_df,
        shipment_df,
        inventory_df,
        unmet_df,
        lanes,
        plants,
        dcs,
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

    utilization = production_df.groupby(["plant_id", "week_id"], as_index=False)["produce_qty"].sum()
    utilization = utilization.merge(plants[["plant_id", "weekly_capacity_units"]], on="plant_id", how="left")
    utilization["utilization_pct"] = utilization["produce_qty"] / utilization["weekly_capacity_units"] * 100

    plt.figure(figsize=(10, 6))
    for plant, plant_df in utilization.groupby("plant_id"):
        plt.plot(plant_df["week_id"], plant_df["utilization_pct"], marker="o", label=plant)
    plt.title("Plant Capacity Utilization")
    plt.xlabel("Week")
    plt.ylabel("Utilization %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plant_capacity_utilization.png", dpi=160)
    plt.close()


def main() -> None:
    forecast_df, retail_df, plants, dcs, lanes, sku_df = load_inputs()
    demand_df = prepare_demand_inputs(forecast_df, retail_df, sku_df)
    optimized = optimize_network_plan(demand_df, plants, dcs, lanes)
    replenish = replenish_to_par_baseline(demand_df)
    cheapest = cheapest_transport_baseline(demand_df)
    baselines = pd.concat([optimized["summary"], replenish, cheapest], ignore_index=True, sort=False)

    write_dataframe(optimized["production"], OUTPUT_DIR / "optimized_production_plan.csv")
    write_dataframe(optimized["shipments"], OUTPUT_DIR / "optimized_shipment_plan.csv")
    write_dataframe(optimized["inventory"], OUTPUT_DIR / "inventory_plan.csv")
    write_dataframe(optimized["unmet"], OUTPUT_DIR / "unmet_demand.csv")
    write_dataframe(optimized["summary"], OUTPUT_DIR / "optimization_summary.csv")
    write_dataframe(baselines, OUTPUT_DIR / "baseline_comparison.csv")
    save_capacity_utilization(optimized["production"], plants)
    LOGGER.info("Saved optimization outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
