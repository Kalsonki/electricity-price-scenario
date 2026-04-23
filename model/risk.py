"""
risk.py – Risk analysis and hedging strategies based on electricity price model results.

Computes from Monte Carlo results:
  - VaR 95%, CVaR 95%, volatility, price spike risk
  - Hedging strategy comparison (spot, fixed, collar, forward, combination)
  - Stress tests (energy crisis, dry winter, nuclear outage, datacenter boom)
  - Optimal hedge recommendation and efficient frontier
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model.scenarios import (
    ScenarioResult, ScenarioParams, SCENARIO_NAMES,
    START_YEAR, END_YEAR, compute_market_adjustments,
)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    """Risk metrics for a single scenario."""
    scenario: str
    var_95: float            # VaR 95% annual price €/MWh
    cvar_95: float           # CVaR 95% annual price €/MWh
    max_monthly_price: float # Highest monthly price (P95 across all months)
    volatility: float        # Monthly price standard deviation (P50-based)
    spike_prob: float        # Estimate: P(some month > 150 €/MWh) per year
    annual_cost_p50: float   # Annual cost at P50 scenario €
    annual_cost_var95: float # Cost at VaR 95% level €
    annual_cost_cvar95: float# Cost at CVaR 95% level €


@dataclass
class HedgeParams:
    """Hedging strategy parameters."""
    strategy: str = "spot"          # spot / fixed_100 / partial_5050 / collar / forward / combination
    fixed_price: float = 65.0       # €/MWh (fixed or forward price)
    floor_price: float = 40.0       # collar floor €/MWh
    cap_price: float = 120.0        # collar cap €/MWh
    hedge_pct: float = 0.70         # forward share (0–1)
    forward_premium: float = 0.05   # forward premium (0–0.15)
    spot_pct: float = 0.30          # combination: spot share (0–1)
    collar_pct: float = 0.40        # combination: collar share (0–1)
    vol_mwh: float = 10_000.0       # annual volume MWh
    dist_type: str = "even"         # even / winter / summer


@dataclass
class HedgeResult:
    """Result for a single hedging strategy."""
    strategy_name: str
    effective_price_p50: float       # €/MWh P50 scenario
    effective_price_p95: float       # €/MWh worst 5%
    annual_cost_p50: float           # €/year P50 scenario
    annual_cost_p95: float           # €/year worst 5%
    hedge_cost_vs_spot_p50: float    # €/year additional cost vs spot P50 (+ = more expensive)
    hedge_benefit_vs_spot_p95: float # €/year savings vs spot P95 (+ = saves)
    risk_reduction_ratio: float      # CVaR improvement % vs spot
    cvar_95: float                   # CVaR 95% cost €/year


@dataclass
class StressTest:
    """A single stress test scenario."""
    name: str
    description: str
    price_spike: float           # peak price €/MWh
    baseline_price: float        # base price level €/MWh
    price_increase_pct: float    # price increase %
    duration_months: int         # estimated duration
    annual_cost_impact: float    # cost impact €/year (at vol_mwh level)
    best_hedge: str              # best hedging strategy


# ── Risk metrics calculation ──────────────────────────────────────────────────

def calculate_risk_metrics(
    scenario_result: ScenarioResult,
    vol_mwh: float = 10_000.0,
) -> RiskMetrics:
    """
    Computes risk metrics for a single scenario from annual_sim_matrix.

    annual_sim_matrix: shape (n_sim, n_years) — annual avg prices per simulation.
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)

    # Annual averages per simulation (average over years)
    avg_annual = sim.mean(axis=1)  # (n_sim,)

    var_95  = float(np.percentile(avg_annual, 95))
    above   = avg_annual[avg_annual >= var_95]
    cvar_95 = float(above.mean()) if len(above) > 0 else var_95

    # Highest monthly price P95 from chart
    mp = scenario_result.monthly_prices
    max_monthly = float(mp["p95"].max()) if "p95" in mp.columns else float(mp["p90"].max())

    # Volatility from P50 monthly series
    volatility = float(mp["p50"].std()) if "p50" in mp.columns else 0.0

    # Price spike risk: P95 value in some month > 150 €/MWh
    p95_col = "p95" if "p95" in mp.columns else "p90"
    spike_months = int((mp[p95_col] > 150.0).sum())
    spike_prob = spike_months / len(mp) if len(mp) > 0 else 0.0

    # Costs at vol_mwh level
    p50_price = float(np.percentile(avg_annual, 50))
    annual_cost_p50   = p50_price   * vol_mwh
    annual_cost_var95 = var_95      * vol_mwh
    annual_cost_cvar95 = cvar_95    * vol_mwh

    return RiskMetrics(
        scenario=scenario_result.name,
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        max_monthly_price=round(max_monthly, 2),
        volatility=round(volatility, 2),
        spike_prob=round(spike_prob, 4),
        annual_cost_p50=round(annual_cost_p50, 0),
        annual_cost_var95=round(annual_cost_var95, 0),
        annual_cost_cvar95=round(annual_cost_cvar95, 0),
    )


# ── Hedging strategy calculation ──────────────────────────────────────────────

def _apply_hedge(prices: np.ndarray, hp: HedgeParams) -> np.ndarray:
    """
    Computes the effective price based on strategy.

    prices: (n_sim,) array of annual average prices €/MWh
    Returns: (n_sim,) effective prices
    """
    s = hp.strategy

    if s == "spot":
        return prices.copy()

    if s == "fixed_100":
        return np.full_like(prices, hp.fixed_price)

    if s == "partial_5050":
        return 0.5 * prices + 0.5 * hp.fixed_price

    if s == "collar":
        return np.clip(prices, hp.floor_price, hp.cap_price)

    if s == "forward":
        # Forward: hedge_pct locked at forward price (based on spot × (1+premium)),
        # remainder at spot price
        fwd = np.mean(prices) * (1.0 + hp.forward_premium)
        return hp.hedge_pct * fwd + (1.0 - hp.hedge_pct) * prices

    if s == "combination":
        forward_pct = max(1.0 - hp.spot_pct - hp.collar_pct, 0.0)
        fwd = np.mean(prices) * (1.0 + hp.forward_premium)
        collar_prices = np.clip(prices, hp.floor_price, hp.cap_price)
        return hp.spot_pct * prices + hp.collar_pct * collar_prices + forward_pct * fwd

    return prices.copy()


def _strategy_label(strategy: str) -> str:
    labels = {
        "spot":          "Full spot",
        "fixed_100":     "Fixed price 100%",
        "partial_5050":  "Partial fixed 50/50",
        "collar":        "Collar strategy",
        "forward":       "Forward hedge 12m rolling",
        "combination":   "Combination strategy",
    }
    return labels.get(strategy, strategy)


def calculate_all_hedges(
    scenario_result: ScenarioResult,
    hedge_params: HedgeParams,
    reference_strategy: str = "spot",
) -> list[HedgeResult]:
    """
    Computes all five standard strategies for comparison.
    Uses the scenario's annual_sim_matrix (annual avg prices).

    Returns a list of HedgeResult objects.
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)
    # Use full period (2025–2035) average per simulation
    avg_annual = sim.mean(axis=1)

    # Spot reference for CVaR
    spot_eff = avg_annual.copy()
    spot_p50  = float(np.percentile(spot_eff, 50))
    spot_p95  = float(np.percentile(spot_eff, 95))
    above_spot = spot_eff[spot_eff >= spot_p95]
    spot_cvar = float(above_spot.mean()) if len(above_spot) > 0 else spot_p95

    results = []
    strategies = ["spot", "fixed_100", "partial_5050", "collar", "forward"]

    for strat in strategies:
        hp_copy = copy.copy(hedge_params)
        hp_copy.strategy = strat
        eff = _apply_hedge(avg_annual, hp_copy)

        p50  = float(np.percentile(eff, 50))
        p95  = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95

        cost_p50 = p50 * hedge_params.vol_mwh
        cost_p95 = p95 * hedge_params.vol_mwh

        hedge_cost = (p50 - spot_p50) * hedge_params.vol_mwh
        hedge_benefit = (spot_p95 - p95) * hedge_params.vol_mwh
        risk_red = (spot_cvar - cvar) / spot_cvar * 100 if spot_cvar > 0 else 0.0

        results.append(HedgeResult(
            strategy_name=_strategy_label(strat),
            effective_price_p50=round(p50, 2),
            effective_price_p95=round(p95, 2),
            annual_cost_p50=round(cost_p50, 0),
            annual_cost_p95=round(cost_p95, 0),
            hedge_cost_vs_spot_p50=round(hedge_cost, 0),
            hedge_benefit_vs_spot_p95=round(hedge_benefit, 0),
            risk_reduction_ratio=round(risk_red, 1),
            cvar_95=round(cvar * hedge_params.vol_mwh, 0),
        ))

    return results


def calculate_active_hedge(
    scenario_result: ScenarioResult,
    hedge_params: HedgeParams,
) -> pd.DataFrame:
    """
    Computes annual effective prices for the selected hedging strategy.

    Returns DataFrame: year, spot_p50, eff_price_p50, eff_price_p10, eff_price_p90
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)
    years = list(range(START_YEAR, END_YEAR + 1))
    rows = []

    for yi, year in enumerate(years):
        annual_col = sim[:, yi]
        eff = _apply_hedge(annual_col, hedge_params)
        rows.append({
            "year": year,
            "spot_p50":      float(np.percentile(annual_col, 50)),
            "eff_price_p10": float(np.percentile(eff, 10)),
            "eff_price_p50": float(np.percentile(eff, 50)),
            "eff_price_p90": float(np.percentile(eff, 90)),
        })

    return pd.DataFrame(rows)


# ── Stress tests ──────────────────────────────────────────────────────────────

def run_stress_tests(base_params: ScenarioParams, vol_mwh: float = 10_000.0) -> list[StressTest]:
    """
    Runs four stress scenarios and returns estimated price impacts.
    Uses deterministic calculations (no full MC).
    """
    STRESS_YEAR = 2027
    BASE_PRICE_BASE = 62.0

    def stressed_price(p: ScenarioParams) -> float:
        return BASE_PRICE_BASE * compute_market_adjustments(p, STRESS_YEAR)

    baseline = stressed_price(base_params)

    results: list[StressTest] = []

    # ── Stress scenario 1: Energy crisis 2021-style ────────────────────────
    p1 = copy.copy(base_params)
    p1.gas_price_mwh = base_params.gas_price_mwh * 3.0   # +200%
    p1.hydro_nordic = "dry"                               # hydro -20%
    p1.electrification_twh = base_params.electrification_twh + 4.25  # consumption +5%
    sp1 = stressed_price(p1)
    # Best hedge: collar (protects against spikes)
    results.append(StressTest(
        name="Energy crisis 2021-style",
        description="Gas price +200%, hydro –25%, consumption +5%",
        price_spike=round(sp1, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp1 / baseline - 1) * 100, 1),
        duration_months=6,
        annual_cost_impact=round((sp1 - baseline) * vol_mwh * (6 / 12), 0),
        best_hedge="Collar strategy (cap 120 €/MWh)",
    ))

    # ── Stress scenario 2: Dry Nordic winter ──────────────────────────────
    p2 = copy.copy(base_params)
    p2.hydro_nordic = "dry"
    # Calm winter → wind power -15%: approximated by reducing capacity
    p2.wind_fi_gw = max(base_params.wind_fi_gw - base_params.wind_fi_gw * 0.15, 0)
    p2.nuclear_se = "one_offline"  # Swedish nuclear partially offline
    sp2 = stressed_price(p2)
    results.append(StressTest(
        name="Dry Nordic winter",
        description="Hydro –30%, wind power –15% (calm winter), SE nuclear restricted",
        price_spike=round(sp2, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp2 / baseline - 1) * 100, 1),
        duration_months=3,
        annual_cost_impact=round((sp2 - baseline) * vol_mwh * (3 / 12), 0),
        best_hedge="Forward hedge 12m rolling (70%)",
    ))

    # ── Stress scenario 3: Nuclear shutdown ───────────────────────────────
    p3 = copy.copy(base_params)
    # OL3 technical fault: simulated as current level + SE reactor offline
    p3.nuclear_fi = "current"
    p3.nuclear_se = "one_offline"
    # Additional capacity simulated lower — using custom multiplier
    sp3 = stressed_price(p3) * 1.12  # +12% additional pressure from OL3 fault
    results.append(StressTest(
        name="Nuclear shutdown",
        description="OL3 technical fault + Swedish reactor offline (6–12 months)",
        price_spike=round(sp3, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp3 / baseline - 1) * 100, 1),
        duration_months=9,
        annual_cost_impact=round((sp3 - baseline) * vol_mwh * (9 / 12), 0),
        best_hedge="Fixed price 100% or forward hedge",
    ))

    # ── Stress scenario 4: Datacenter boom ────────────────────────────────
    p4 = copy.copy(base_params)
    p4.datacenter_growth_pct = 45.0   # explosive growth
    p4.datacenter_base_twh = base_params.datacenter_base_twh
    p4.electrification_twh = base_params.electrification_twh + 20.0  # +20 TWh
    sp4 = stressed_price(p4)
    results.append(StressTest(
        name="Datacenter boom",
        description="Datacenter consumption +20 TWh unexpectedly fast (2027–2029)",
        price_spike=round(sp4, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp4 / baseline - 1) * 100, 1),
        duration_months=36,
        annual_cost_impact=round((sp4 - baseline) * vol_mwh, 0),
        best_hedge="Collar strategy or long-term forward hedge",
    ))

    return results


# ── Optimal hedge recommendation ─────────────────────────────────────────────

def get_hedge_recommendation(
    risk_metrics: dict[str, RiskMetrics],
    hedge_results: list[HedgeResult] | None = None,
) -> dict[str, Any]:
    """
    Analyzes risk level and gives a hedging recommendation.

    Returns: {risk_class, recommendation_text, best_strategy}
    """
    base = risk_metrics.get("base")
    if base is None:
        return {"risk_class": "unknown", "recommendation_text": "Insufficient data.", "best_strategy": "spot"}

    # Classify risk level
    if base.volatility < 12.0:
        risk_class = "low"
    elif base.volatility < 22.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    # Find best strategy by CVaR minimization
    best = "collar"
    if hedge_results:
        sorted_h = sorted(hedge_results, key=lambda h: h.cvar_95)
        if sorted_h:
            best = sorted_h[0].strategy_name

    if risk_class == "low":
        text = (
            f"Low-risk market conditions (volatility {base.volatility:.1f} €/MWh). "
            "Recommended: 70% spot + 30% forward hedge. "
            "Full spot is a reasonable option, but a short forward hedge provides protection against surprises."
        )
    elif risk_class == "moderate":
        text = (
            f"Moderate risk level (volatility {base.volatility:.1f} €/MWh, "
            f"CVaR 95% = {base.cvar_95:.1f} €/MWh). "
            "Recommended: Collar strategy (floor 40 €/MWh, cap 120 €/MWh) "
            "or 50/50 fixed + spot. Balances cost and risk."
        )
    else:
        text = (
            f"High risk level (volatility {base.volatility:.1f} €/MWh, "
            f"CVaR 95% = {base.cvar_95:.1f} €/MWh, "
            f"price spike risk {base.spike_prob*100:.1f}%). "
            "Recommended: Collar strategy or fixed price 70–100%. "
            "CVaR-minimizing best strategy: "
            f"{best}."
        )

    return {
        "risk_class": risk_class,
        "recommendation_text": text,
        "best_strategy": best,
        "volatility": base.volatility,
        "cvar_95": base.cvar_95,
    }


# ── Efficient frontier ────────────────────────────────────────────────────────

def compute_efficient_frontier(
    scenario_result: ScenarioResult,
    vol_mwh: float,
    fixed_price: float = 65.0,
    floor_price: float = 40.0,
    cap_price: float = 120.0,
    forward_premium: float = 0.05,
) -> pd.DataFrame:
    """
    Computes efficient frontier chart data:
    x = additional hedge cost vs spot P50 (€/year)
    y = CVaR 95% (€/year)

    Varies hedge_pct 0..100% and collar cap 80..200 €/MWh.
    Returns DataFrame: strategy, hedge_cost, cvar_95
    """
    sim = scenario_result.annual_sim_matrix
    avg_annual = sim.mean(axis=1)

    spot_p50 = float(np.percentile(avg_annual, 50))
    spot_p95 = float(np.percentile(avg_annual, 95))
    above_spot = avg_annual[avg_annual >= spot_p95]
    spot_cvar = float(above_spot.mean()) if len(above_spot) > 0 else spot_p95

    rows = []

    base_hp = HedgeParams(
        fixed_price=fixed_price,
        floor_price=floor_price,
        cap_price=cap_price,
        forward_premium=forward_premium,
        vol_mwh=vol_mwh,
    )

    # Fixed price at various shares
    for pct in np.linspace(0, 1.0, 21):
        hp = copy.copy(base_hp)
        hp.strategy = "fixed_100" if pct == 1.0 else "spot" if pct == 0.0 else "partial_5050"
        if hp.strategy == "partial_5050":
            # Emulate partial hedge at pct%
            eff = pct * fixed_price + (1 - pct) * avg_annual
        elif hp.strategy == "fixed_100":
            eff = np.full_like(avg_annual, fixed_price)
        else:
            eff = avg_annual.copy()
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategy": f"Fixed {pct*100:.0f}%",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "type": "Fixed",
        })

    # Collar at various cap levels
    for cap in np.arange(80, 210, 20):
        eff = np.clip(avg_annual, floor_price, float(cap))
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategy": f"Collar cap={cap:.0f}",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "type": "Collar",
        })

    # Forward hedge at various hedge shares
    fwd = float(np.mean(avg_annual)) * (1.0 + forward_premium)
    for pct in np.linspace(0, 1.0, 11):
        eff = pct * fwd + (1 - pct) * avg_annual
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategy": f"Forward {pct*100:.0f}%",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "type": "Forward",
        })

    return pd.DataFrame(rows)


def build_risk_metrics_table(risk_metrics: dict[str, RiskMetrics]) -> pd.DataFrame:
    """Builds risk metrics table for three scenarios for comparison."""
    rows = []
    labels = {"low": "Low", "base": "Base", "high": "High"}
    for sc in ["low", "base", "high"]:
        m = risk_metrics.get(sc)
        if m is None:
            continue
        rows.append({
            "Scenario": labels.get(sc, sc),
            "VaR 95% (€/MWh)": m.var_95,
            "CVaR 95% (€/MWh)": m.cvar_95,
            "Max Monthly Price (€/MWh)": m.max_monthly_price,
            "Volatility (€/MWh)": m.volatility,
            "Price Spike Risk (%)": round(m.spike_prob * 100, 1),
        })
    return pd.DataFrame(rows)
