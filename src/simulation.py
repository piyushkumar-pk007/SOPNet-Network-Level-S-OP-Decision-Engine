from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import simpy

from src.config import OUTPUT_DIR, SIMULATION_CONFIG
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_inputs(
    demand_path: Path | None = None,
    shipment_path: Path | None = None,
    unmet_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand_df = pd.read_csv(demand_path or OUTPUT_DIR / "reconciled_forecast.csv", parse_dates=["date"])
    shipment_df = pd.read_csv(shipment_path or OUTPUT_DIR / "optimized_shipment_plan.csv")
    unmet_file = unmet_path or OUTPUT_DIR / "unmet_demand.csv"
    if unmet_file.exists() and unmet_file.stat().st_size > 0:
        try:
            unmet_df = pd.read_csv(unmet_file)
        except pd.errors.EmptyDataError:
            unmet_df = pd.DataFrame()
    else:
        unmet_df = pd.DataFrame()
    return demand_df, shipment_df, unmet_df


def _prepare_simulation_base(demand_df: pd.DataFrame, shipment_df: pd.DataFrame) -> pd.DataFrame:
    demand = demand_df[
        (demand_df["method"] == "simplified_mint") & (demand_df["series_level"] == "store_category")
    ].copy()
    if demand.empty:
        demand = demand_df[demand_df["series_level"].str.contains("store_category", na=False)].copy()
    demand["retail_node_id"] = demand["store_id"] + "__" + demand["cat_id"]
    planned = (
        shipment_df[shipment_df["lane_type"] == "dc_to_retail"]
        .groupby(["destination_id", "category", "week_id"], as_index=False)["shipment_qty"]
        .sum()
        .rename(columns={"destination_id": "retail_node_id", "category": "cat_id", "shipment_qty": "planned_supply"})
    )
    base = demand.merge(planned, on=["retail_node_id", "cat_id", "week_id"], how="left")
    base["planned_supply"] = base["planned_supply"].fillna(0)
    return base


def _single_replication(base_df: pd.DataFrame, replication_id: int) -> Dict[str, float]:
    env = simpy.Environment()
    metrics: Dict[str, float] = {}

    def weekly_process(environment: simpy.Environment) -> None:
        service_levels: List[float] = []
        stockout_units: List[float] = []
        ending_inventory: List[float] = []
        for week_id in sorted(base_df["week_id"].unique()):
            week_df = base_df[base_df["week_id"] == week_id]
            for row in week_df.itertuples():
                demand = max(
                    0.0,
                    np.random.normal(
                        loc=row.forecast,
                        scale=max(abs(row.forecast) * SIMULATION_CONFIG.demand_std_pct, 1.0),
                    ),
                )
                lead_time_factor = 1.0
                if np.random.rand() < SIMULATION_CONFIG.lead_time_delay_probability:
                    lead_time_factor -= SIMULATION_CONFIG.lead_time_delay_pct * 0.2
                if np.random.rand() < SIMULATION_CONFIG.supply_disruption_probability:
                    lead_time_factor -= 0.15
                realized_supply = max(row.planned_supply * lead_time_factor, 0.0)
                service = min(realized_supply / demand, 1.0) if demand > 0 else 1.0
                stockout = max(demand - realized_supply, 0.0)
                ending_inv = max(realized_supply - demand, 0.0)
                service_levels.append(service)
                stockout_units.append(stockout)
                ending_inventory.append(ending_inv)
            yield environment.timeout(1)

        metrics.update(
            {
                "replication_id": replication_id,
                "avg_service_level": float(np.mean(service_levels)) if service_levels else 1.0,
                "stockout_units": float(np.sum(stockout_units)),
                "expected_stockout_cost": float(np.sum(stockout_units) * 12.0),
                "avg_ending_inventory": float(np.mean(ending_inventory)) if ending_inventory else 0.0,
            }
        )

    env.process(weekly_process(env))
    env.run()
    return metrics


def run_simulation(base_df: pd.DataFrame, replications: int = SIMULATION_CONFIG.replications) -> pd.DataFrame:
    return pd.DataFrame([_single_replication(base_df, replication_id=i + 1) for i in range(replications)])


def main() -> None:
    demand_df, shipment_df, _ = load_inputs()
    base_df = _prepare_simulation_base(demand_df, shipment_df)
    sim_results = run_simulation(base_df, SIMULATION_CONFIG.replications)

    summary = pd.DataFrame(
        [
            {
                "avg_service_level": sim_results["avg_service_level"].mean(),
                "p5_service_level": sim_results["avg_service_level"].quantile(0.05),
                "avg_stockout_units": sim_results["stockout_units"].mean(),
                "expected_stockout_cost": sim_results["expected_stockout_cost"].mean(),
                "avg_ending_inventory": sim_results["avg_ending_inventory"].mean(),
                "service_level_ci_low": sim_results["avg_service_level"].quantile(0.025),
                "service_level_ci_high": sim_results["avg_service_level"].quantile(0.975),
            }
        ]
    )
    service_dist = sim_results[["replication_id", "avg_service_level"]].rename(columns={"avg_service_level": "service_level"})
    stockout_dist = sim_results[["replication_id", "stockout_units"]]

    write_dataframe(sim_results, OUTPUT_DIR / "simulation_results.csv")
    write_dataframe(service_dist, OUTPUT_DIR / "service_level_distribution.csv")
    write_dataframe(stockout_dist, OUTPUT_DIR / "stockout_distribution.csv")
    write_dataframe(summary, OUTPUT_DIR / "simulation_summary.csv")
    LOGGER.info("Saved simulation outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
