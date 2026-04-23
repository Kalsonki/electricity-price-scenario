"""
costs.py – Company electricity cost calculation based on scenario results.
"""

import numpy as np
import pandas as pd
from typing import Optional

from model.scenarios import ScenarioResult, SCENARIO_NAMES


def build_consumption_profile(
    annual_mwh: float,
    distribution: str,
    custom_weights: Optional[dict[int, float]] = None,
) -> dict[int, float]:
    """
    Builds a monthly consumption profile (MWh per month).

    distribution: 'even' | 'winter' | 'summer' | 'custom'
    custom_weights: {month: %-share} if distribution=='custom'
    """
    base = annual_mwh / 12.0

    if distribution == "even":
        return {m: base for m in range(1, 13)}

    if distribution == "winter":
        winter = {11, 12, 1, 2}
        weights = {m: (1.5 if m in winter else 1.0) for m in range(1, 13)}
        total = sum(weights.values())
        return {m: annual_mwh * weights[m] / total for m in range(1, 13)}

    if distribution == "summer":
        summer = {6, 7, 8}
        weights = {m: (1.5 if m in summer else 1.0) for m in range(1, 13)}
        total = sum(weights.values())
        return {m: annual_mwh * weights[m] / total for m in range(1, 13)}

    if distribution == "custom" and custom_weights:
        total_pct = sum(custom_weights.values()) or 1.0
        return {m: annual_mwh * (custom_weights.get(m, 0) / total_pct) for m in range(1, 13)}

    return {m: base for m in range(1, 13)}


def apply_contract_price(
    spot_price: float,
    contract_type: str,
    fixed_share: float = 0.0,
    fixed_price: float = 0.0,
) -> float:
    """
    Computes the effective price based on contract type.

    contract_type: 'spot' | 'partial_fixed' | 'fixed'
    fixed_share: 0–1 fixed share (in partial fixed)
    fixed_price: fixed price component €/MWh
    """
    if contract_type == "fixed":
        return fixed_price
    if contract_type == "partial_fixed":
        return (1.0 - fixed_share) * spot_price + fixed_share * fixed_price
    return spot_price  # spot


def calculate_costs(
    scenario_results: dict[str, ScenarioResult],
    annual_mwh: float,
    distribution: str,
    contract_type: str,
    fixed_share: float = 0.0,
    fixed_price: float = 0.0,
    custom_weights: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Computes monthly costs for all scenarios.

    Returns DataFrame with columns:
    scenario, year, month, consumption_mwh,
    price_p10, price_p50, price_p90,
    eff_price_p10, eff_price_p50, eff_price_p90,
    cost_p10, cost_p50, cost_p90
    """
    profile = build_consumption_profile(annual_mwh, distribution, custom_weights)
    records = []

    for scenario_name, result in scenario_results.items():
        for _, row in result.monthly_prices.iterrows():
            month = int(row["month"])
            consumption = profile.get(month, annual_mwh / 12)

            eff_p10 = apply_contract_price(row["p10"], contract_type, fixed_share, fixed_price)
            eff_p50 = apply_contract_price(row["p50"], contract_type, fixed_share, fixed_price)
            eff_p90 = apply_contract_price(row["p90"], contract_type, fixed_share, fixed_price)

            records.append({
                "scenario": scenario_name,
                "year": int(row["year"]),
                "month": month,
                "consumption_mwh": consumption,
                "price_p10": row["p10"],
                "price_p50": row["p50"],
                "price_p90": row["p90"],
                "eff_price_p10": eff_p10,
                "eff_price_p50": eff_p50,
                "eff_price_p90": eff_p90,
                "cost_p10": consumption * eff_p10,
                "cost_p50": consumption * eff_p50,
                "cost_p90": consumption * eff_p90,
            })

    return pd.DataFrame(records)


def annual_costs(cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates monthly costs to annual level.
    Returns: scenario, year, cost_p10, cost_p50, cost_p90
    """
    return (
        cost_df.groupby(["scenario", "year"])[["cost_p10", "cost_p50", "cost_p90"]]
        .sum()
        .reset_index()
    )


def cumulative_costs(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Computes cumulative costs annually per scenario."""
    frames = []
    for scenario in SCENARIO_NAMES:
        sub = annual_df[annual_df["scenario"] == scenario].sort_values("year").copy()
        sub["cum_p10"] = sub["cost_p10"].cumsum()
        sub["cum_p50"] = sub["cost_p50"].cumsum()
        sub["cum_p90"] = sub["cost_p90"].cumsum()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def risk_exposure(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Computes risk exposure: High P50 minus Low P50 per year."""
    high = annual_df[annual_df["scenario"] == "high"][["year", "cost_p50"]].rename(
        columns={"cost_p50": "high"}
    )
    low = annual_df[annual_df["scenario"] == "low"][["year", "cost_p50"]].rename(
        columns={"cost_p50": "low"}
    )
    merged = high.merge(low, on="year")
    merged["risk_eur"] = merged["high"] - merged["low"]
    return merged


def optimization_savings(
    scenario_results: dict[str, ScenarioResult],
    annual_mwh: float,
    shift_fraction: float = 0.10,
    scenario_name: str = "base",
) -> dict[str, float]:
    """
    Computes savings potential by shifting winter consumption to summer.

    shift_fraction: fraction of winter shifted (default 10%)
    Returns: shifted_mwh, savings_eur_year, winter_price, summer_price
    """
    result = scenario_results.get(scenario_name)
    if result is None:
        return {"shifted_mwh": 0.0, "savings_eur_year": 0.0, "winter_price": 0.0, "summer_price": 0.0}

    monthly = result.monthly_prices.copy()

    # Use year 2027 median prices
    ref_year = 2027
    ref_rows = monthly[monthly["year"] == ref_year]
    if ref_rows.empty:
        ref_rows = monthly

    ref = ref_rows.set_index("month")["p50"]

    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]

    winter_price = float(ref[[m for m in winter_months if m in ref.index]].mean()) if any(m in ref.index for m in winter_months) else 70.0
    summer_price  = float(ref[[m for m in summer_months if m in ref.index]].mean()) if any(m in ref.index for m in summer_months) else 50.0

    shifted_mwh = annual_mwh * shift_fraction * (3 / 12)
    price_diff = winter_price - summer_price
    savings = shifted_mwh * price_diff

    return {
        "shifted_mwh":    round(shifted_mwh, 1),
        "savings_eur_year": round(max(savings, 0.0), 0),
        "winter_price":    round(winter_price, 1),
        "summer_price":    round(summer_price, 1),
    }
