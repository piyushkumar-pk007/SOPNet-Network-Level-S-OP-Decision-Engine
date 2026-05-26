from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def _node(node_id: str, parent_id: str, level: int, name: str, key: str) -> dict[str, str | int]:
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "level": level,
        "node_name": name,
        "aggregation_key": key,
    }


def load_weekly_demand(path: Path | None = None) -> pd.DataFrame:
    weekly_path = path or PROCESSED_DIR / "weekly_demand.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly demand file at {weekly_path}. Run preprocessing first.")
    return pd.read_csv(weekly_path, parse_dates=["date"])


def build_hierarchy_table(weekly_df: pd.DataFrame) -> pd.DataFrame:
    rows = [_node("TOTAL_NETWORK", "", 0, "Total Network", "total_network")]

    for state_id in sorted(weekly_df["state_id"].dropna().unique()):
        state_node = f"STATE::{state_id}"
        rows.append(_node(state_node, "TOTAL_NETWORK", 1, state_id, state_id))
        state_df = weekly_df[weekly_df["state_id"] == state_id]
        for store_id in sorted(state_df["store_id"].dropna().unique()):
            store_node = f"STORE::{store_id}"
            rows.append(_node(store_node, state_node, 2, store_id, store_id))
            store_df = state_df[state_df["store_id"] == store_id]
            for cat_id in sorted(store_df["cat_id"].dropna().unique()):
                cat_node = f"CATEGORY::{store_id}::{cat_id}"
                rows.append(_node(cat_node, store_node, 3, cat_id, f"{store_id}|{cat_id}"))
                cat_df = store_df[store_df["cat_id"] == cat_id]
                for dept_id in sorted(cat_df["dept_id"].dropna().unique()):
                    dept_node = f"DEPARTMENT::{store_id}::{dept_id}"
                    rows.append(_node(dept_node, cat_node, 4, dept_id, f"{store_id}|{dept_id}"))
                    dept_df = cat_df[cat_df["dept_id"] == dept_id]
                    for item_id in sorted(dept_df["item_id"].dropna().unique()):
                        item_node = f"ITEM::{store_id}::{item_id}"
                        rows.append(_node(item_node, dept_node, 5, item_id, f"{store_id}|{item_id}"))
    return pd.DataFrame(rows).drop_duplicates()


def _aggregate_level(weekly_df: pd.DataFrame, level: int, group_cols: list[str], prefix: str) -> pd.DataFrame:
    level_df = weekly_df.groupby(group_cols + ["week_id"], as_index=False)["weekly_demand"].sum()
    level_df["level"] = level
    level_df["node_id"] = level_df[group_cols].astype(str).agg("::".join, axis=1).map(lambda value: f"{prefix}::{value}")
    return level_df


def build_hierarchical_weekly_demand(weekly_df: pd.DataFrame) -> pd.DataFrame:
    frames = [weekly_df.groupby("week_id", as_index=False)["weekly_demand"].sum().assign(level=0, node_id="TOTAL_NETWORK")]

    for level, group_cols, prefix in [
        (1, ["state_id"], "STATE"),
        (2, ["store_id"], "STORE"),
        (3, ["store_id", "cat_id"], "CATEGORY"),
        (4, ["store_id", "dept_id"], "DEPARTMENT"),
        (5, ["store_id", "item_id"], "ITEM"),
    ]:
        frames.append(_aggregate_level(weekly_df, level, group_cols, prefix))

    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    weekly = load_weekly_demand()
    hierarchy_table = build_hierarchy_table(weekly)
    hierarchical_demand = build_hierarchical_weekly_demand(weekly)

    write_dataframe(hierarchy_table, PROCESSED_DIR / "hierarchy_table.csv")
    write_dataframe(hierarchical_demand, PROCESSED_DIR / "hierarchical_weekly_demand.csv")
    LOGGER.info("Saved hierarchy outputs to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
