from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_M5_DIR = DATA_DIR / "raw" / "m5"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_NETWORK_DIR = DATA_DIR / "synthetic_network"
OUTPUT_DIR = DATA_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"


@dataclass
class SamplingConfig:
    selected_stores: list[str] = field(default_factory=lambda: ["CA_1", "TX_1", "WI_1"])
    top_departments: int = 5
    max_skus: int = 120
    last_n_days: int = 728
    forecast_horizon_weeks: int = 4
    optional_horizon_weeks: int = 8
    optimization_aggregation_level: str = "category_store_week"


@dataclass
class OptimizationConfig:
    planning_horizon_weeks: int = 4
    big_m: float = 1_000_000.0
    stockout_penalty_per_unit: float = 12.0
    default_min_run_units: float = 100.0
    initial_inventory_cover_weeks: float = 0.4
    service_levels: dict[str, float] = field(
        default_factory=lambda: {"A": 0.95, "B": 0.90, "C": 0.85}
    )


@dataclass
class SimulationConfig:
    replications: int = 100
    max_replications: int = 500
    demand_std_pct: float = 0.15
    lead_time_delay_probability: float = 0.15
    lead_time_delay_pct: float = 0.30
    supply_disruption_probability: float = 0.05
    random_seed: int = 42


@dataclass
class ScenarioConfig:
    demand_shock_pct: float = 0.20
    lead_time_shock_pct: float = 0.30
    transport_cost_shock_pct: float = 0.15
    plant_capacity_shock_pct: float = 0.25


SAMPLING_CONFIG = SamplingConfig()
OPTIMIZATION_CONFIG = OptimizationConfig()
SIMULATION_CONFIG = SimulationConfig()
SCENARIO_CONFIG = ScenarioConfig()
