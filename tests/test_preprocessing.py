import pandas as pd

from src.preprocessing import build_weekly_demand, melt_sales_to_long


def test_melt_sales_to_long_creates_long_rows():
    sales = pd.DataFrame(
        {
            "id": ["A"],
            "item_id": ["ITEM_1"],
            "dept_id": ["DEPT_1"],
            "cat_id": ["CAT_1"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [2],
            "d_2": [3],
        }
    )
    result = melt_sales_to_long(sales)
    assert len(result) == 2
    assert set(result["d"]) == {"d_1", "d_2"}


def test_build_weekly_demand_aggregates_and_creates_revenue():
    long_df = pd.DataFrame(
        {
            "id": ["A", "A"],
            "item_id": ["ITEM_1", "ITEM_1"],
            "dept_id": ["DEPT_1", "DEPT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "d": ["d_1", "d_2"],
            "daily_demand": [2, 3],
        }
    )
    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "wm_yr_wk": [1001, 1001],
            "event_name_1": ["No Event", "Holiday"],
            "event_type_1": ["No Event", "Cultural"],
            "snap_CA": [0, 1],
            "snap_TX": [0, 0],
            "snap_WI": [0, 0],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["ITEM_1"],
            "wm_yr_wk": [1001],
            "sell_price": [4.0],
        }
    )
    weekly = build_weekly_demand(long_df, calendar, prices)
    assert weekly["weekly_demand"].iloc[0] == 5
    assert weekly["revenue"].iloc[0] == 20

