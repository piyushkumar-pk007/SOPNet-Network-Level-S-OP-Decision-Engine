from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROCESSED_DIR, RAW_M5_DIR, SYNTHETIC_NETWORK_DIR


def ensure_directories() -> None:
    for directory in (RAW_M5_DIR, PROCESSED_DIR, SYNTHETIC_NETWORK_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(OUTPUT_DIR / "pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def snake_case(text: str) -> str:
    text = text.strip().replace("%", "pct").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.lower().strip("_")


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def weighted_average(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    wts = np.asarray(list(weights), dtype=float)
    if vals.size == 0 or np.nansum(wts) == 0:
        return float(np.nan)
    return float(np.average(vals, weights=wts))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom * 100) if denom else np.nan


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def latest_week_ids(df: pd.DataFrame, week_col: str, n: int) -> list:
    return sorted(df[week_col].dropna().unique())[-n:]
