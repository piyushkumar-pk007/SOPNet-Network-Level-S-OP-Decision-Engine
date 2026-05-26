import pandas as pd

from src.hierarchy_builder import build_hierarchy_table, build_hierarchical_weekly_demand


def test_build_hierarchy_table_contains_total_and_item_levels():
    weekly = pd.DataFrame(
        {
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["DEPT_1", "DEPT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "week_id": [1001, 1001],
            "weekly_demand": [10, 20],
        }
    )
    hierarchy = build_hierarchy_table(weekly)
    assert "TOTAL_NETWORK" in hierarchy["node_id"].values
    assert (hierarchy["level"] == 5).any()


def test_hierarchical_weekly_demand_rolls_up_values():
    weekly = pd.DataFrame(
        {
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["DEPT_1", "DEPT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "week_id": [1001, 1001],
            "weekly_demand": [10, 20],
        }
    )
    rolled = build_hierarchical_weekly_demand(weekly)
    total = rolled.loc[rolled["node_id"] == "TOTAL_NETWORK", "weekly_demand"].iloc[0]
    assert total == 30

