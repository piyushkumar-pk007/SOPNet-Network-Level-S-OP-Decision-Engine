from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_weekly_demand(path: Path | None = None) -> pd.DataFrame:
    weekly_path = path or PROCESSED_DIR / "weekly_demand.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly demand file at {weekly_path}. Run preprocessing first.")
    return pd.read_csv(weekly_path, parse_dates=["date"])


def build_hierarchy_table(weekly_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "node_id": "TOTAL_NETWORK",
            "parent_id": "",
            "level": 0,
            "node_name": "Total Network",
            "aggregation_key": "total_network",
        }
    ]

    for state_id in sorted(weekly_df["state_id"].dropna().unique()):
        state_node = f"STATE::{state_id}"
        rows.append({"node_id": state_node, "parent_id": "TOTAL_NETWORK", "level": 1, "node_name": state_id, "aggregation_key": state_id})
        state_subset = weekly_df[weekly_df["state_id"] == state_id]
        for store_id in sorted(state_subset["store_id"].dropna().unique()):
            store_node = f"STORE::{store_id}"
            rows.append({"node_id": store_node, "parent_id": state_node, "level": 2, "node_name": store_id, "aggregation_key": store_id})
            store_subset = state_subset[state_subset["store_id"] == store_id]
            for cat_id in sorted(store_subset["cat_id"].dropna().unique()):
                cat_node = f"CATEGORY::{store_id}::{cat_id}"
                rows.append({"node_id": cat_node, "parent_id": store_node, "level": 3, "node_name": cat_id, "aggregation_key": f"{store_id}|{cat_id}"})
                cat_subset = store_subset[store_subset["cat_id"] == cat_id]
                for dept_id in sorted(cat_subset["dept_id"].dropna().unique()):
                    dept_node = f"DEPARTMENT::{store_id}::{dept_id}"
                    rows.append({"node_id": dept_node, "parent_id": cat_node, "level": 4, "node_name": dept_id, "aggregation_key": f"{store_id}|{dept_id}"})
                    dept_subset = cat_subset[cat_subset["dept_id"] == dept_id]
                    for item_id in sorted(dept_subset["item_id"].dropna().unique()):
                        item_node = f"ITEM::{store_id}::{item_id}"
                        rows.append({"node_id": item_node, "parent_id": dept_node, "level": 5, "node_name": item_id, "aggregation_key": f"{store_id}|{item_id}"})
    return pd.DataFrame(rows).drop_duplicates()


def build_hierarchical_weekly_demand(weekly_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    total = weekly_df.groupby("week_id", as_index=False)["weekly_demand"].sum()
    total["level"] = 0
    total["node_id"] = "TOTAL_NETWORK"
    frames.append(total)

    for level, group_cols, prefix in [
        (1, ["state_id"], "STATE"),
        (2, ["store_id"], "STORE"),
        (3, ["store_id", "cat_id"], "CATEGORY"),
        (4, ["store_id", "dept_id"], "DEPARTMENT"),
        (5, ["store_id", "item_id"], "ITEM"),
    ]:
        frame = (
            weekly_df.groupby(group_cols + ["week_id"], as_index=False)["weekly_demand"]
            .sum()
        )
        frame["level"] = level
        frame["node_id"] = frame[group_cols].astype(str).agg("::".join, axis=1).map(lambda x: f"{prefix}::{x}")
        frames.append(frame)

    hierarchical = pd.concat(frames, ignore_index=True, sort=False)
    return hierarchical


def main() -> None:
    weekly_df = load_weekly_demand()
    hierarchy_table = build_hierarchy_table(weekly_df)
    hierarchical_demand = build_hierarchical_weekly_demand(weekly_df)

    write_dataframe(hierarchy_table, PROCESSED_DIR / "hierarchy_table.csv")
    write_dataframe(hierarchical_demand, PROCESSED_DIR / "hierarchical_weekly_demand.csv")
    LOGGER.info("Saved hierarchy outputs to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()

