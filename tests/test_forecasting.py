import pandas as pd

from src.forecasting import build_forecast_base


def test_build_forecast_base_creates_series_id():
    weekly = pd.DataFrame(
        {
            "state_id": ["CA", "CA"],
            "store_id": ["CA_1", "CA_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "week_id": [1001, 1002],
            "date": pd.to_datetime(["2024-01-07", "2024-01-14"]),
            "month": [1, 1],
            "quarter": [1, 1],
            "year": [2024, 2024],
            "week_of_year": [1, 2],
            "weekly_demand": [10, 15],
            "avg_sell_price": [2.0, 2.2],
            "is_event_week": [0, 1],
        }
    )
    base = build_forecast_base(weekly)
    assert "series_id" in base.columns
    assert base["series_level"].iloc[0] == "store_category"

