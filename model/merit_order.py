"""
merit_order.py – Merit order model for the Finnish electricity market.

Marginal cost order:
Wind → Solar → Nuclear → Hydro → CHP → Import → Gas
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class MeritOrderParams:
    gas_price_mwh: float = 40.0      # €/MWh
    co2_price_t: float   = 70.0      # €/t
    water_level: str     = "normal"  # normal / dry / wet
    nordpool_ref: float  = 55.0      # Nord Pool reference price €/MWh
    month: int           = 1

# Marginal costs €/MWh per generation source
MARGINAL_BASE = {
    "wind":    (0.0,  3.0),
    "solar":   (0.0,  2.0),
    "nuclear": (8.0, 12.0),
    "import":  None,  # calculated as nordpool_ref + 2
    "chp":     None,  # calculated with gas formula
    "gas":     None,  # calculated with gas formula
}

# Hydro opportunity cost €/MWh per reservoir level
HYDRO_OC = {
    "wet":    (0.0,  2.0),
    "normal": (0.0,  5.0),
    "dry":    (15.0, 40.0),
}

# Seasonal factor for hydro cost
HYDRO_SEASONAL = {
    1:1.75, 2:1.80, 3:1.60, 4:0.40, 5:0.35, 6:0.85,
    7:0.90, 8:0.95, 9:0.75, 10:0.70, 11:0.65, 12:1.70
}

def _hydro_marginal(water_level: str, month: int) -> float:
    """Computes the hydro marginal cost €/MWh."""
    lo, hi = HYDRO_OC.get(water_level, (0.0, 5.0))
    base = (lo + hi) / 2.0
    seasonal = HYDRO_SEASONAL.get(month, 1.0)
    mult = {"wet": 0.4, "normal": 1.0, "dry": 1.8}.get(water_level, 1.0)
    return round(base * seasonal * mult, 2)

def chp_marginal(gas_price_mwh: float, co2_price_t: float) -> float:
    """
    CHP plant marginal cost €/MWh electricity.
    Efficiency ~0.85 (combined heat+power), CO₂ ~0.20 t/MWh gas.
    Formula: (gas_price + CO₂_factor × CO₂_price) / efficiency
    """
    eta = 0.85                          # CHP efficiency (electricity + heat)
    co2_per_mwh_gas = 0.20             # t CO₂ per MWh natural gas
    fuel_cost = gas_price_mwh + co2_per_mwh_gas * co2_price_t
    return round(fuel_cost / eta, 2)

# Internal alias (compatibility)
_chp_marginal = chp_marginal

def gas_marginal(gas_price_mwh: float, co2_price_t: float) -> float:
    """
    Gas turbine marginal cost €/MWh electricity.
    Efficiency ~0.55 (CCGT), CO₂ ~0.20 t/MWh gas.
    Formula: (gas_price + CO₂_factor × CO₂_price) / efficiency
    """
    eta = 0.55                          # CCGT efficiency
    co2_per_mwh_gas = 0.20             # t CO₂ per MWh natural gas
    fuel_cost = gas_price_mwh + co2_per_mwh_gas * co2_price_t
    return round(fuel_cost / eta, 2)

# Internal alias (compatibility)
_gas_marginal = gas_marginal

@dataclass
class MeritOrderSlice:
    """One capacity slice on the merit order curve."""
    source: str
    capacity_mw: float
    marginal_cost: float
    cumulative_mw: float = 0.0

def build_merit_order(
    params: MeritOrderParams,
    capacity_mw: Dict[str, float],
) -> List[MeritOrderSlice]:
    """
    Builds the merit order curve for the given capacity dictionary.

    capacity_mw: {"wind": X, "solar": Y, "nuclear": Z,
                  "hydro": V, "chp": C, "import": T, "gas": G}
    Returns a sorted list of MeritOrderSlice objects.
    """
    hydro_mc  = _hydro_marginal(params.water_level, params.month)
    chp_mc    = _chp_marginal(params.gas_price_mwh, params.co2_price_t)
    import_mc = params.nordpool_ref + 2.0
    gas_mc    = _gas_marginal(params.gas_price_mwh, params.co2_price_t)

    source_costs = {
        "wind":    2.0,
        "solar":   1.5,
        "nuclear": 10.0,
        "hydro":   hydro_mc,
        "chp":     chp_mc,
        "import":  import_mc,
        "gas":     gas_mc,
    }

    slices = []
    for source, cap in capacity_mw.items():
        if cap > 0:
            slices.append(MeritOrderSlice(
                source=source,
                capacity_mw=float(cap),
                marginal_cost=source_costs.get(source, 50.0),
            ))

    slices.sort(key=lambda s: s.marginal_cost)

    cumulative = 0.0
    for s in slices:
        cumulative += s.capacity_mw
        s.cumulative_mw = round(cumulative, 0)

    return slices

def calculate_market_price(
    month: int,
    capacity_mw: Dict[str, float],
    demand_mw: float,
    water_level: str = "normal",
    gas_price: float = 40.0,
    co2_price: float = 70.0,
    nordpool_ref: float = 55.0,
) -> Tuple[float, str, float]:
    """
    Computes the market price using the merit order model.

    Returns: (market_price_eur_mwh, marginal_source, capacity_surplus_mw)
    """
    params = MeritOrderParams(
        gas_price_mwh=gas_price,
        co2_price_t=co2_price,
        water_level=water_level,
        nordpool_ref=nordpool_ref,
        month=month,
    )
    slices = build_merit_order(params, capacity_mw)

    if not slices:
        return nordpool_ref, "unknown", 0.0

    # Find marginal generation source
    marginal_source = slices[-1].source
    marginal_cost   = slices[-1].marginal_cost
    total_cap       = slices[-1].cumulative_mw

    # If demand exceeds supply, price = last + premium
    surplus_mw = total_cap - demand_mw
    if surplus_mw < 0:
        # Overload → premium
        premium = abs(surplus_mw) / demand_mw * 50.0
        price = marginal_cost + premium
    else:
        # Find the correct marginal cost
        price = marginal_cost
        for s in slices:
            if s.cumulative_mw >= demand_mw:
                price = s.marginal_cost
                marginal_source = s.source
                break

    return round(price, 2), marginal_source, round(surplus_mw, 0)

def merit_order_time_series(
    months: range,
    base_capacity: Dict[str, float],
    demand_mw: float,
    gas_price: float = 40.0,
    co2_price: float = 70.0,
    water_level: str = "normal",
) -> pd.DataFrame:
    """Computes the merit order price for each month."""
    records = []
    for month in months:
        price, source, surplus = calculate_market_price(
            month, base_capacity, demand_mw, water_level, gas_price, co2_price
        )
        records.append({
            "month": month,
            "price_eur_mwh": price,
            "marginal_source": source,
            "surplus_mw": surplus,
        })
    return pd.DataFrame(records)

def merit_order_to_df(slices: List[MeritOrderSlice]) -> pd.DataFrame:
    """Converts a list of MeritOrderSlice objects to a DataFrame for charting."""
    return pd.DataFrame([{
        "source": s.source,
        "capacity_mw": s.capacity_mw,
        "marginal_cost": s.marginal_cost,
        "cumulative_mw": s.cumulative_mw,
    } for s in slices])

SOURCE_COLORS = {
    "wind":    "#2196F3",
    "solar":   "#FFC107",
    "nuclear": "#9C27B0",
    "hydro":   "#00BCD4",
    "chp":     "#FF9800",
    "import":  "#4CAF50",
    "gas":     "#F44336",
}
