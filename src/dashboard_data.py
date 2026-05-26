from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR, PROCESSED_DIR, SYNTHETIC_NETWORK_DIR


def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_dashboard_datasets() -> dict[str, pd.DataFrame]:
    return {
        "weekly_demand": _safe_read(PROCESSED_DIR / "weekly_demand.csv", parse_dates=["date"]),
        "forecast_results": _safe_read(OUTPUT_DIR / "forecast_results.csv", parse_dates=["date"]),
        "model_comparison": _safe_read(OUTPUT_DIR / "model_comparison.csv"),
        "reconciled_forecast": _safe_read(OUTPUT_DIR / "reconciled_forecast.csv", parse_dates=["date"]),
        "optimization_summary": _safe_read(OUTPUT_DIR / "optimization_summary.csv"),
        "baseline_comparison": _safe_read(OUTPUT_DIR / "baseline_comparison.csv"),
        "production_plan": _safe_read(OUTPUT_DIR / "optimized_production_plan.csv"),
        "shipment_plan": _safe_read(OUTPUT_DIR / "optimized_shipment_plan.csv"),
        "inventory_plan": _safe_read(OUTPUT_DIR / "inventory_plan.csv"),
        "unmet_demand": _safe_read(OUTPUT_DIR / "unmet_demand.csv"),
        "simulation_results": _safe_read(OUTPUT_DIR / "simulation_results.csv"),
        "simulation_summary": _safe_read(OUTPUT_DIR / "simulation_summary.csv"),
        "scenario_comparison": _safe_read(OUTPUT_DIR / "scenario_comparison.csv"),
        "plants": _safe_read(SYNTHETIC_NETWORK_DIR / "plants.csv"),
        "retail_nodes": _safe_read(SYNTHETIC_NETWORK_DIR / "retail_nodes.csv"),
    }


def executive_metrics(datasets: dict[str, pd.DataFrame]) -> dict[str, float]:
    forecast = datasets["forecast_results"]
    summary = datasets["optimization_summary"]
    baseline = datasets["baseline_comparison"]
    simulation = datasets["simulation_summary"]
    production = datasets["production_plan"]
    plants = datasets["plants"]

    total_forecast_demand = float(forecast["forecast"].sum()) if not forecast.empty else 0.0
    optimized_cost = float(summary["total_cost"].iloc[0]) if not summary.empty else 0.0
    expected_service = float(simulation["avg_service_level"].iloc[0]) if not simulation.empty else 0.0
    stockout_risk = float(simulation["avg_stockout_units"].iloc[0]) if not simulation.empty else 0.0

    if not baseline.empty and "strategy" in baseline.columns and "total_cost" in baseline.columns:
        baseline_cost = baseline[baseline["strategy"] == "replenish_to_par"]["total_cost"]
        saving = float(baseline_cost.iloc[0] - optimized_cost) if not baseline_cost.empty else 0.0
    else:
        saving = 0.0

    if not production.empty and not plants.empty:
        plant_utilization = (
            production.groupby("plant_id", as_index=False)["produce_qty"].sum()
            .merge(plants[["plant_id", "weekly_capacity_units"]], on="plant_id", how="left")
        )
        plant_utilization["utilization"] = plant_utilization["produce_qty"] / plant_utilization["weekly_capacity_units"]
        top_constrained = plant_utilization.sort_values("utilization", ascending=False)["plant_id"].iloc[0]
    else:
        top_constrained = ""

    return {
        "total_forecast_demand": total_forecast_demand,
        "optimized_total_cost": optimized_cost,
        "expected_service_level": expected_service,
        "stockout_risk": stockout_risk,
        "cost_saving_vs_baseline": saving,
        "top_constrained_plant": top_constrained,
    }
