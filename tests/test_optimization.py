import pandas as pd

from src.optimization import optimize_network_plan


def test_optimize_network_plan_returns_summary():
    demand = pd.DataFrame(
        {
            "retail_node_id": ["CA_1__CAT_1"],
            "store_id": ["CA_1"],
            "cat_id": ["CAT_1"],
            "week_id": [1001],
            "required_demand": [500.0],
            "abc_class": ["A"],
            "service_level_target": [0.95],
        }
    )
    plants = pd.DataFrame(
        {
            "plant_id": ["Plant_A"],
            "weekly_capacity_units": [1000.0],
            "fixed_setup_cost": [100.0],
            "variable_production_cost": [1.0],
            "product_categories_supported": ["CAT_1"],
        }
    )
    dcs = pd.DataFrame(
        {
            "dc_id": ["DC_West"],
            "storage_capacity_units": [1000.0],
            "holding_cost_per_unit": [0.1],
        }
    )
    lanes = pd.DataFrame(
        {
            "source_id": ["Plant_A", "DC_West"],
            "destination_id": ["DC_West", "CA_1__CAT_1"],
            "lane_type": ["plant_to_dc", "dc_to_retail"],
            "transport_cost_per_unit": [0.2, 0.1],
            "lead_time_days": [3, 2],
            "max_lane_capacity": [1000.0, 1000.0],
        }
    )
    result = optimize_network_plan(demand, plants, dcs, lanes)
    assert not result["summary"].empty
    assert result["summary"]["service_level"].iloc[0] >= 0.95

