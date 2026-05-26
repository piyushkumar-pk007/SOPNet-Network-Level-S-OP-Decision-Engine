from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import PROCESSED_DIR, RAW_M5_DIR, SAMPLING_CONFIG
from src.utils import ensure_directories, get_logger, write_dataframe


LOGGER = get_logger(__name__)
REQUIRED_FILES = [
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
]
PUBLIC_M5_ZIP_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _extract_archives() -> None:
    for archive_path in RAW_M5_DIR.glob("*.zip"):
        LOGGER.info("Extracting archive %s", archive_path.name)
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(RAW_M5_DIR)


def _download_from_public_mirror() -> None:
    target_zip = RAW_M5_DIR / "m5-forecasting-accuracy-public.zip"
    LOGGER.info("Downloading M5 files from public mirror: %s", PUBLIC_M5_ZIP_URL)
    with urllib.request.urlopen(PUBLIC_M5_ZIP_URL) as response, open(target_zip, "wb") as fh:
        shutil.copyfileobj(response, fh)
    _extract_archives()


def _download_from_kaggle() -> None:
    load_dotenv(dotenv_path=ENV_PATH)
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    competition = os.getenv("KAGGLE_COMPETITION", "m5-forecasting-accuracy")

    if not username or not key:
        LOGGER.info("Kaggle credentials not found. Using manual file mode.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        LOGGER.warning("Kaggle package unavailable. Using manual file mode.")
        return

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle credentials were not loaded successfully. "
            "Check that .env exists in the project root and contains valid KAGGLE_USERNAME and KAGGLE_KEY values."
        ) from exc
    LOGGER.info("Downloading Kaggle competition files for %s", competition)
    try:
        api.competition_download_files(competition, path=str(RAW_M5_DIR), force=False, quiet=False)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "Unauthorized" in message:
            LOGGER.warning(
                "Kaggle authentication worked, but competition file access was denied. "
                "Falling back to a public M5 mirror."
            )
            _download_from_public_mirror()
            return
        raise
    _extract_archives()


def validate_required_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for filename in REQUIRED_FILES:
        path = RAW_M5_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {filename}. Place M5 files inside data/raw/m5/ or configure Kaggle credentials."
            )
        files[filename] = path
    return files


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error if last_error else RuntimeError(f"Could not read {path.name}")


def sample_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    day_cols = [col for col in sales_df.columns if col.startswith("d_")]
    if len(day_cols) > SAMPLING_CONFIG.last_n_days:
        keep_day_cols = day_cols[-SAMPLING_CONFIG.last_n_days :]
    else:
        keep_day_cols = day_cols

    store_filtered = sales_df[sales_df["store_id"].isin(SAMPLING_CONFIG.selected_stores)].copy()
    dept_rank = (
        store_filtered.groupby("dept_id")[keep_day_cols]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    selected_depts = dept_rank.head(SAMPLING_CONFIG.top_departments).index.tolist()
    dept_filtered = store_filtered[store_filtered["dept_id"].isin(selected_depts)].copy()

    sku_rank = dept_filtered.groupby("item_id")[keep_day_cols].sum().sum(axis=1).sort_values(ascending=False)
    selected_skus = sku_rank.head(SAMPLING_CONFIG.max_skus).index.tolist()
    sampled = dept_filtered[dept_filtered["item_id"].isin(selected_skus)].copy()

    base_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]
    return sampled[base_columns + keep_day_cols]


def clean_calendar_data(calendar_df: pd.DataFrame) -> pd.DataFrame:
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"], errors="coerce")
    calendar_df["event_name_1"] = calendar_df["event_name_1"].fillna("No Event")
    calendar_df["event_type_1"] = calendar_df["event_type_1"].fillna("No Event")
    return calendar_df


def clean_prices(prices_df: pd.DataFrame, sampled_sales_df: pd.DataFrame) -> pd.DataFrame:
    allowed_items = sampled_sales_df["item_id"].unique()
    allowed_stores = sampled_sales_df["store_id"].unique()
    return prices_df[
        prices_df["item_id"].isin(allowed_items) & prices_df["store_id"].isin(allowed_stores)
    ].copy()


def main() -> None:
    ensure_directories()
    if not all((RAW_M5_DIR / filename).exists() for filename in REQUIRED_FILES):
        _download_from_kaggle()

    files = validate_required_files()
    sales_df = _safe_read_csv(files["sales_train_validation.csv"], low_memory=False)
    calendar_df = _safe_read_csv(files["calendar.csv"], low_memory=False)
    prices_df = _safe_read_csv(files["sell_prices.csv"], low_memory=False)

    sales_sample = sample_sales_data(sales_df)
    calendar_clean = clean_calendar_data(calendar_df)
    prices_clean = clean_prices(prices_df, sales_sample)

    write_dataframe(sales_sample, PROCESSED_DIR / "sales_sample.csv")
    write_dataframe(calendar_clean, PROCESSED_DIR / "calendar_clean.csv")
    write_dataframe(prices_clean, PROCESSED_DIR / "prices_clean.csv")
    LOGGER.info("Saved sampled ingestion outputs to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
