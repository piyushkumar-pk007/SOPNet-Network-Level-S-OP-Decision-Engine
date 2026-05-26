import pandas as pd

from src.simulation import run_simulation


def test_run_simulation_returns_replications():
    base = pd.DataFrame(
        {
            "week_id": [1001, 1002],
            "retail_node_id": ["CA_1__CAT_1", "CA_1__CAT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "forecast": [100.0, 120.0],
            "planned_supply": [105.0, 125.0],
        }
    )
    results = run_simulation(base, replications=5)
    assert len(results) == 5
    assert "avg_service_level" in results.columns

