"""
capacity.py – Finland electricity generation capacity model by month.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

# Nuclear power MW
NUCLEAR_BASE_MW = 900      # OL1+OL2
OL3_MW = 1600
NUCLEAR_UTILIZATION = 0.90

# Wind capacity factors per season
WIND_CF = {1:0.35, 2:0.35, 3:0.35, 4:0.28, 5:0.28, 6:0.22,
           7:0.22, 8:0.22, 9:0.32, 10:0.32, 11:0.32, 12:0.35}

# Solar capacity factors
SOLAR_CF = {1:0.02, 2:0.04, 3:0.08, 4:0.12, 5:0.16, 6:0.18,
            7:0.17, 8:0.15, 9:0.10, 10:0.05, 11:0.02, 12:0.01}

# CHP + hydro MW
HYDRO_CHP_MW = {1:2000, 2:2000, 3:1800, 4:2200, 5:2500, 6:1200,
                7:800, 8:600, 9:800, 10:1200, 11:1800, 12:2000}

@dataclass
class CapacityParams:
    nuclear_fi: str = "current"        # current / ol3_hanhikivi / new_plant / smr
    wind_fi_total_gw: float = 7.0      # total installed GW (base + additions)
    solar_fi_gw: float = 1.5
    interconnect_fi_se_mw: float = 2200
    interconnect_fi_ee_mw: float = 1000

NUCLEAR_OPTIONS_MW = {
    "current":       NUCLEAR_BASE_MW + OL3_MW,          # 2500
    "ol3_hanhikivi": NUCLEAR_BASE_MW + OL3_MW + 1000,   # 3500
    "new_plant":     NUCLEAR_BASE_MW + OL3_MW + 1600,   # 4100
    "smr":           NUCLEAR_BASE_MW + OL3_MW + 500,    # 3000
}

def calculate_monthly_capacity(params: CapacityParams, month: int) -> dict:
    """
    Computes monthly capacity per generation source (MW).
    Returns dict: nuclear, wind, solar, hydro_chp, total, interconnect
    """
    nuclear_mw  = NUCLEAR_OPTIONS_MW.get(params.nuclear_fi, 2500) * NUCLEAR_UTILIZATION
    wind_mw     = params.wind_fi_total_gw * 1000 * WIND_CF.get(month, 0.28)
    solar_mw    = params.solar_fi_gw * 1000 * SOLAR_CF.get(month, 0.05)
    hydro_chp   = HYDRO_CHP_MW.get(month, 1200)
    interconnect = params.interconnect_fi_se_mw + params.interconnect_fi_ee_mw

    total = nuclear_mw + wind_mw + solar_mw + hydro_chp

    return {
        "nuclear_mw":   round(nuclear_mw, 0),
        "wind_mw":      round(wind_mw, 0),
        "solar_mw":     round(solar_mw, 0),
        "hydro_chp_mw": round(hydro_chp, 0),
        "total_mw":     round(total, 0),
        "interconnect_mw": round(interconnect, 0),
    }

def calculate_capacity_margin(cap: dict, demand_mw: float) -> dict:
    """
    Computes capacity surplus/deficit and price premium.
    demand_mw: estimated peak hour demand MW
    """
    surplus_mw = cap["total_mw"] + cap["interconnect_mw"] - demand_mw
    congestion = surplus_mw < 0
    # Premium if interconnections are congested
    premium = max(-surplus_mw / demand_mw * 50.0, 0.0) if congestion else 0.0
    utilization = min(demand_mw / (cap["total_mw"] + cap["interconnect_mw"] + 1), 1.0)

    return {
        "surplus_mw":   round(surplus_mw, 0),
        "congestion":   congestion,
        "premium_eur":  round(premium, 1),
        "utilization":  round(utilization, 3),
    }

def capacity_time_series(params: CapacityParams, start_year: int = 2025, end_year: int = 2035) -> pd.DataFrame:
    """Computes the capacity time series for all year-month combinations."""
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cap = calculate_monthly_capacity(params, month)
            records.append({"year": year, "month": month, **cap})
    return pd.DataFrame(records)

def find_critical_months(params: CapacityParams, demand_twh_annual: float = 88.0,
                          top_n: int = 5) -> pd.DataFrame:
    """
    Identifies top_n risk months in 2025–2035.
    Criteria: high consumption + low generation + full interconnection capacity.
    """
    # Monthly consumption TWh → MW peak demand (approximately TWh*1000*1.6/720)
    monthly_demand_factor = {
        1:1.35, 2:1.30, 3:1.10, 4:0.95, 5:0.85, 6:0.80,
        7:0.82, 8:0.88, 9:0.95, 10:1.05, 11:1.22, 12:1.38
    }
    avg_demand_mw = demand_twh_annual * 1e6 / 8760  # MW

    records = []
    for year in range(2025, 2036):
        for month in range(1, 13):
            cap = calculate_monthly_capacity(params, month)
            demand_mw = avg_demand_mw * monthly_demand_factor.get(month, 1.0) * 1.6
            margin = calculate_capacity_margin(cap, demand_mw)
            if margin["congestion"]:
                records.append({
                    "year": year, "month": month,
                    "surplus_mw": margin["surplus_mw"],
                    "utilization_pct": round(margin["utilization"] * 100, 1),
                    "premium_eur_mwh": margin["premium_eur"],
                })

    if not records:
        return pd.DataFrame(columns=["year","month","surplus_mw","utilization_pct","premium_eur_mwh"])

    df = pd.DataFrame(records)
    return df.nsmallest(top_n, "surplus_mw").reset_index(drop=True)
