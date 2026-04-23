"""
data_fetch.py – Data loading from Excel files and synthetic fallback.

Main functions:
  - load_fundamental_data(filepath)  → standardized DataFrame from Excel file
  - load_historical_prices()         → monthly prices (synthetic)
  - generate_synthetic_prices()      → synthetic historical data
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.data_inspect import inspect_excel, paras_valilehti

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Standard column names and their detection keywords
_COLUMN_MAP = {
    "date":               ["date", "pvm", "päivä", "aika", "time", "timestamp", "period", "vuosi", "year", "month"],
    "price_fi":           ["hinta", "price", "spot", "electricity", "sahkö", "sähkö", "eur_mwh", "€/mwh"],
    "consumption":        ["kulutus", "consumption", "demand", "käyttö", "load"],
    "wind_capacity":      ["tuulivoima", "wind", "tuuli", "wind_cap"],
    "hydro_production":   ["vesivoima", "hydro", "vesisähkö", "hydro_prod"],
    "nuclear_production": ["ydinvoima", "nuclear", "ydin", "nuclear_prod"],
    "gas_price":          ["kaasu", "gas", "lng", "ttf", "nbp", "gas_price"],
    "co2_price":          ["co2", "hiili", "carbon", "päästö", "emission", "eua"],
}


def _best_column_match(columns: list[str], keywords: list[str]) -> str | None:
    """Finds the best matching column for the given keywords (case-insensitive)."""
    for col in columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    """
    Attempts to convert a series to datetime using multiple methods.
    Returns the original series if conversion fails.
    """
    for fmt in [None, "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%m/%d/%Y"]:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            continue
    # Try ymd numeric series (e.g. 20150101)
    try:
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="raise")
    except Exception:
        pass
    return series


def _normalize_to_monthly(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Normalizes a time series to monthly level.

    If data is already monthly, returns as-is.
    If daily or hourly, aggregates to monthly averages.
    """
    df = df.copy()
    df[date_col] = _to_datetime_safe(df[date_col])

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        logger.warning("Date column could not be converted: %s", date_col)
        return df

    df["_year"] = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month

    # Detect frequency: if more than 13 rows per year, data is finer than monthly
    rows_per_year = df.groupby("_year").size().median()
    if rows_per_year <= 13:
        # Already monthly or coarser — use as-is
        df["date"] = pd.to_datetime(df[["_year", "_month"]].assign(day=1))
        df = df.drop(columns=[date_col, "_year", "_month"], errors="ignore")
        return df

    # Aggregate numeric columns to monthly averages
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("_year", "_month")]
    agg = df.groupby(["_year", "_month"])[numeric_cols].mean().reset_index()
    agg["date"] = pd.to_datetime(agg[["_year", "_month"]].rename(columns={"_year": "year", "_month": "month"}).assign(day=1))
    agg = agg.drop(columns=["_year", "_month"], errors="ignore")
    return agg


def load_fundamental_data(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Loads fundamental data from an Excel file and standardizes columns.

    Steps:
    1. Calls inspect_excel() to detect structure
    2. Selects the best sheet
    3. Loads and normalizes data
    4. Renames columns to standard names
    5. Handles missing values

    Returns:
        (df, meta)
        df:   DataFrame with standard columns (date, price_fi, consumption, ...)
        meta: dict with inspect result + found columns + assumptions
    """
    filepath = Path(filepath)
    meta: dict[str, Any] = {"assumptions": [], "found_columns": {}, "inspect": {}}

    # Step 1: inspect file
    inspect = inspect_excel(filepath)
    meta["inspect"] = inspect

    if "error" in inspect:
        logger.error("Failed to open Excel file: %s", inspect["error"])
        return pd.DataFrame(), meta

    best_sheet = paras_valilehti(inspect)
    if best_sheet is None:
        logger.error("No suitable sheet found")
        return pd.DataFrame(), meta

    meta["used_sheet"] = best_sheet
    sheet_info = inspect["sheets"][best_sheet]
    columns = sheet_info["columns"]

    # Step 2: load sheet
    try:
        raw = pd.read_excel(filepath, sheet_name=best_sheet, engine="openpyxl")
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        raw.columns = raw.columns.astype(str)
    except Exception as e:
        logger.error("Failed to load sheet: %s", e)
        meta["assumptions"].append(f"Sheet loading failed: {e}")
        return pd.DataFrame(), meta

    raw_columns = list(raw.columns)

    # Step 3: detect columns
    found: dict[str, str] = {}  # standard_name -> original_column_name
    for std_name, keywords in _COLUMN_MAP.items():
        match = _best_column_match(raw_columns, keywords)
        if match:
            found[std_name] = match

    meta["found_columns"] = {k: v for k, v in found.items() if k != "date"}

    # Step 4: date — required field or generated
    date_col = found.get("date")
    if date_col and date_col in raw.columns:
        df = _normalize_to_monthly(raw, date_col)
    else:
        # Try to find separate year+month columns
        year_col = _best_column_match(raw_columns, ["year", "vuosi"])
        month_col = _best_column_match(raw_columns, ["month", "kuukausi"])
        if year_col and month_col:
            raw["_date_gen"] = pd.to_datetime(
                raw[[year_col, month_col]].rename(columns={year_col: "year", month_col: "month"}).assign(day=1)
            )
            df = raw.copy()
            df["date"] = df["_date_gen"]
            df = df.drop(columns=["_date_gen"], errors="ignore")
            meta["assumptions"].append("Date built from year+month columns.")
        else:
            logger.warning("No date column found — generating monthly sequence")
            df = raw.copy()
            df["date"] = pd.date_range(start="2015-01", periods=len(df), freq="MS")
            meta["assumptions"].append(
                "No date column found. Using automatic monthly sequence starting 2015-01."
            )

    # Step 5: rename detected columns to standard names
    rename_map: dict[str, str] = {}
    for std_name, orig_col in found.items():
        if std_name == "date":
            continue
        if orig_col in df.columns:
            rename_map[orig_col] = std_name

    df = df.rename(columns=rename_map)

    # Report missing standard columns as assumptions
    all_std = [k for k in _COLUMN_MAP if k != "date"]
    for std in all_std:
        if std not in df.columns:
            meta["assumptions"].append(
                f"'{std}' not found in data — using synthetic default in model."
            )

    # Step 6: ensure date column and sort
    if "date" not in df.columns:
        meta["assumptions"].append("Could not create date column.")
        return df, meta

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Step 7: convert numeric data and interpolate missing values
    for col in df.select_dtypes(include="object").columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    logger.info(
        "Fundamental data loaded: %d rows, columns: %s",
        len(df), list(df.columns)
    )
    return df, meta


# ── Synthetic historical data ─────────────────────────────────────────────────

def generate_synthetic_prices(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """
    Generates realistic synthetic monthly price data for years 2015–2025.

    Models actual Finnish price levels:
    - 2015–2020: 30–45 €/MWh, normal seasonal variation
    - 2021–2022: energy crisis, price spikes
    - 2023–2024: normalization with OL3
    - 2025: low price level, abundant hydro + OL3
    """
    rng = np.random.default_rng(42)

    annual_base = {
        2015: 31.0, 2016: 33.0, 2017: 37.0, 2018: 44.0, 2019: 40.0,
        2020: 28.0, 2021: 72.0, 2022: 140.0, 2023: 55.0, 2024: 45.0,
        2025: 38.0,  # low price level: OL3 in full production, abundant hydro
    }
    month_factors = {
        1: 1.35, 2: 1.30, 3: 1.10, 4: 0.95, 5: 0.85, 6: 0.80,
        7: 0.82, 8: 0.88, 9: 0.95, 10: 1.05, 11: 1.20, 12: 1.38,
    }
    crisis_boost = {
        (2021, 12): 2.5, (2022, 1): 3.2, (2022, 2): 2.8,
        (2022, 8): 3.5, (2022, 9): 4.0, (2022, 10): 3.0,
        (2022, 11): 2.5, (2022, 12): 2.2,
    }
    # 2025 monthly realized prices (Nord Pool FI wholesale price €/MWh)
    # Source: Energia.fi 2025 annual review, Yle, Vaasan Sähkö market review
    # Annual average ~41 €/MWh (down 9% from 2024)
    price_2025_override = {
        1:  55.0,  # January: cold winter, high consumption
        2:  48.0,  # February
        3:  32.0,  # March: prices started to fall
        4:  18.0,  # April: abundant hydro, very cheap
        5:  22.0,  # May: flood peak
        6:  20.0,  # June: ~21 €/MWh (source: Vaasan Sähkö)
        7:  22.0,  # July: ~21 €/MWh
        8:  45.0,  # August: prices rose (heat wave, low wind)
        9:  42.0,  # September
        10: 52.0,  # October: autumn-winter begins
        11: 58.0,  # November
        12: 62.0,  # December: cold, high consumption
    }

    records = []
    for year in range(start_year, end_year + 1):
        base = annual_base.get(year, 45.0)
        for month in range(1, 13):
            if year == 2025 and month in price_2025_override:
                price = price_2025_override[month] * rng.normal(1.0, 0.03)
            else:
                mf = month_factors[month]
                boost = crisis_boost.get((year, month), 1.0)
                noise = rng.normal(1.0, 0.08)
                price = max(base * mf * boost * noise, 0.0)
            records.append({"year": year, "month": month, "price_eur_mwh": round(price, 2)})

    return pd.DataFrame(records)


def load_historical_prices() -> pd.DataFrame:
    """
    Loads historical price data.

    Uses synthetic data (realistic fallback without API key).
    Returns DataFrame with columns: year, month, price_eur_mwh
    """
    logger.info("Using synthetic historical data")
    return generate_synthetic_prices()
