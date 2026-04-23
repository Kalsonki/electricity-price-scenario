"""
data_inspect.py – Automatic detection and structure analysis of Excel files.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Keyword lists for detection
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    "date", "pvm", "päivä", "aika", "time", "vuosi", "year",
    "kuukausi", "month", "timestamp", "period",
]

_PRICE_PATTERNS = [
    "hinta", "price", "spot", "€", "eur", "mwh", "kwh",
    "cost", "rate", "tariff", "electricity",
]

_CONSUMPTION_PATTERNS = [
    "kulutus", "consumption", "demand", "käyttö", "use",
    "load", "energy_use", "kulutus_mwh",
]

_CAPACITY_PATTERNS = [
    "kapasiteetti", "capacity", "mw", "gw", "installed",
    "teho", "power", "cap",
]

_PRODUCTION_PATTERNS = [
    "tuulivoima", "wind", "hydro", "nuclear", "ydinvoima",
    "vesivoima", "solar", "aurinko", "tuuli", "production",
    "tuotanto", "generation",
]

_GAS_PATTERNS = ["kaasu", "gas", "lng", "ttf", "nbp"]

_CO2_PATTERNS = ["co2", "hiili", "carbon", "päästö", "emission", "eua"]


def _match_patterns(col: str, patterns: list[str]) -> bool:
    """Checks whether any keyword is found in the column name."""
    col_lower = col.lower()
    return any(p in col_lower for p in patterns)


def _detect_column_roles(columns: list[str]) -> dict[str, list[str]]:
    """
    Groups columns by role (date, price, consumption, etc.).

    Returns dict: role -> list of column names.
    """
    roles: dict[str, list[str]] = {
        "date": [],
        "price": [],
        "consumption": [],
        "capacity": [],
        "production": [],
        "gas": [],
        "co2": [],
        "other": [],
    }
    for col in columns:
        if _match_patterns(col, _DATE_PATTERNS):
            roles["date"].append(col)
        elif _match_patterns(col, _CO2_PATTERNS):
            roles["co2"].append(col)
        elif _match_patterns(col, _GAS_PATTERNS):
            roles["gas"].append(col)
        elif _match_patterns(col, _PRICE_PATTERNS):
            roles["price"].append(col)
        elif _match_patterns(col, _CONSUMPTION_PATTERNS):
            roles["consumption"].append(col)
        elif _match_patterns(col, _PRODUCTION_PATTERNS):
            roles["production"].append(col)
        elif _match_patterns(col, _CAPACITY_PATTERNS):
            roles["capacity"].append(col)
        else:
            roles["other"].append(col)
    return roles


def inspect_excel(filepath: str | Path) -> dict[str, Any]:
    """
    Analyzes the structure of an Excel file without loading data.

    Returns a dictionary containing for each sheet:
      - sheet_names: all sheet names
      - sheets: dict of sheets containing:
          * columns: column names
          * dtypes: column data types
          * n_rows: row count
          * preview: first 3 rows (dict list)
          * roles: automatically detected column roles
          * detected_data: human-readable summary

    On error, returns {'error': str, 'sheets': {}}.
    """
    filepath = Path(filepath)
    result: dict[str, Any] = {"filename": filepath.name, "sheets": {}}

    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
    except Exception as e:
        return {"error": f"Could not open file: {e}", "sheets": {}}

    result["sheet_names"] = xl.sheet_names

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception as e:
            result["sheets"][sheet] = {"error": str(e)}
            continue

        # Remove fully empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        columns = list(df.columns.astype(str))
        roles = _detect_column_roles(columns)

        # Detected data summary
        detected: list[str] = []
        if roles["date"]:
            detected.append(f"Time series: {', '.join(roles['date'])}")
        if roles["price"]:
            detected.append(f"Price data: {', '.join(roles['price'])}")
        if roles["consumption"]:
            detected.append(f"Consumption data: {', '.join(roles['consumption'])}")
        if roles["production"]:
            detected.append(f"Production data: {', '.join(roles['production'])}")
        if roles["capacity"]:
            detected.append(f"Capacity data: {', '.join(roles['capacity'])}")
        if roles["gas"]:
            detected.append(f"Gas price: {', '.join(roles['gas'])}")
        if roles["co2"]:
            detected.append(f"CO2 emission allowance: {', '.join(roles['co2'])}")

        result["sheets"][sheet] = {
            "columns": columns,
            "dtypes": {c: str(t) for c, t in zip(df.columns.astype(str), df.dtypes)},
            "n_rows": len(df),
            "preview": df.head(3).astype(str).to_dict(orient="records"),
            "roles": roles,
            "detected_data": detected if detected else ["Not detected automatically"],
        }

    return result


def paras_valilehti(inspect_result: dict[str, Any]) -> str | None:
    """
    Selects the best sheet for data loading.

    Prioritizes the sheet with the most recognized columns.
    Returns None if the result has no sheets.
    """
    sheets = inspect_result.get("sheets", {})
    if not sheets:
        return None

    def score(sheet_info: dict) -> int:
        roles = sheet_info.get("roles", {})
        return sum(
            len(v)
            for k, v in roles.items()
            if k != "other" and isinstance(v, list)
        )

    return max(sheets.keys(), key=lambda s: score(sheets[s]))
