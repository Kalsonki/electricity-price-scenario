"""
charts.py – Plotly charts for electricity price scenarios, market dynamics, and risk analysis.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.scenarios import (
    RegressionResult, ScenarioResult,
    SCENARIO_LABELS, SCENARIO_COLORS, SCENARIO_NAMES,
    START_YEAR, END_YEAR,
)

MONTH_LABELS_SHORT = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
MONTH_LABELS_FULL = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May",     6: "June",     7: "July",   8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

_LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=80, b=60),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a hex color code (#RRGGBB) to rgba format for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _date_label(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS CHARTS (Excel fundamental data)
# ══════════════════════════════════════════════════════════════════════════════

def fundamental_time_series(fundamental_df: pd.DataFrame) -> go.Figure:
    """Time series chart of detected fundamental data."""
    numeric_cols = [c for c in fundamental_df.columns if c != "date"
                    and pd.api.types.is_numeric_dtype(fundamental_df[c])]
    if not numeric_cols or "date" not in fundamental_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available to plot", **_LAYOUT_BASE)
        return fig

    priority = ["price_fi", "gas_price", "co2_price", "consumption",
                "wind_capacity", "hydro_production", "nuclear_production"]
    ordered = [c for c in priority if c in numeric_cols]
    ordered += [c for c in numeric_cols if c not in ordered]

    col_labels = {
        "price_fi":           "Electricity spot price (€/MWh)",
        "consumption":        "Consumption (MWh)",
        "wind_capacity":      "Wind power (MW)",
        "hydro_production":   "Hydro power (MWh)",
        "nuclear_production": "Nuclear power (MWh)",
        "gas_price":          "Gas price",
        "co2_price":          "CO₂ emission allowance",
    }

    fig = go.Figure()
    colors_cycle = ["#1565C0", "#2E7D32", "#B71C1C", "#F57C00", "#6A1B9A", "#00796B", "#AD1457"]

    for i, col in enumerate(ordered[:6]):
        label = col_labels.get(col, col)
        fig.add_trace(go.Scatter(
            x=fundamental_df["date"],
            y=fundamental_df[col],
            name=label,
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.8),
            hovertemplate=f"%{{x|%Y-%m}}: %{{y:.2f}}<extra>{label}</extra>",
        ))

    fig.update_layout(
        title="Historical fundamental data",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def correlation_heatmap(fundamental_df: pd.DataFrame) -> go.Figure:
    """Correlation matrix: which variables explain prices most."""
    numeric_cols = [c for c in fundamental_df.columns if c != "date"
                    and pd.api.types.is_numeric_dtype(fundamental_df[c])]
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.update_layout(title="Too few columns for correlation matrix", **_LAYOUT_BASE)
        return fig

    col_labels = {
        "price_fi":           "Spot price",
        "consumption":        "Consumption",
        "wind_capacity":      "Wind power",
        "hydro_production":   "Hydro power",
        "nuclear_production": "Nuclear power",
        "gas_price":          "Gas price",
        "co2_price":          "CO₂",
    }

    corr = fundamental_df[numeric_cols].corr()
    labels = [col_labels.get(c, c) for c in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="Correlation matrix – relationships between variables",
        height=420,
        margin=dict(l=120, r=20, t=80, b=100),
    )
    return fig


def regression_coef_chart(regression: RegressionResult) -> go.Figure:
    """Bar chart of regression model coefficients."""
    col_labels = {
        "wind_capacity":      "Wind power",
        "hydro_production":   "Hydro power",
        "nuclear_production": "Nuclear power",
        "gas_price":          "Gas price",
        "co2_price":          "CO₂",
        "month_sin":          "Seasonal trend (sin)",
        "month_cos":          "Seasonal trend (cos)",
    }

    if not regression.coef:
        fig = go.Figure()
        fig.update_layout(title="No regression coefficients available", **_LAYOUT_BASE)
        return fig

    features = list(regression.coef.keys())
    coefs = list(regression.coef.values())
    labels = [col_labels.get(f, f) for f in features]
    colors = ["#2E7D32" if c >= 0 else "#B71C1C" for c in coefs]

    fig = go.Figure(go.Bar(
        x=labels,
        y=coefs,
        marker_color=colors,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Regression model coefficients (R² = {regression.r2:.3f})",
        xaxis_title="Feature",
        yaxis_title="Coefficient (standardized)",
        height=380,
        yaxis=dict(gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#555"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PRICE SCENARIO CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def price_scenario_chart(
    scenario_results: dict[str, ScenarioResult],
    historical_df: pd.DataFrame,
    visible_scenarios: Optional[list[str]] = None,
) -> go.Figure:
    """
    Line chart: three scenario paths 2025–2035 + historical comparison.
    Shaded area = P10–P90 uncertainty band.
    """
    if visible_scenarios is None:
        visible_scenarios = SCENARIO_NAMES

    fig = go.Figure()

    if not historical_df.empty:
        hist = historical_df[historical_df["year"] >= 2023].sort_values(["year", "month"])
        hist["label"] = hist.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        fig.add_trace(go.Scatter(
            x=hist["label"],
            y=hist["price_eur_mwh"],
            name="Historical (2023–2025)",
            line=dict(color="#757575", width=1.5, dash="dot"),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Historical</extra>",
        ))

    for scenario in SCENARIO_NAMES:
        if scenario not in visible_scenarios:
            continue
        result = scenario_results[scenario]
        df = result.monthly_prices.sort_values(["year", "month"])
        df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        color = result.color

        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p90"], df["p10"].iloc[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{result.label} (range)",
        ))

        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df["p50"],
            name=result.label,
            line=dict(color=color, width=2),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + result.label + "</extra>",
        ))

    CHART_START = 2023  # Show historical data from this year
    fig.update_layout(
        title="Electricity price scenario paths 2023–2038",
        xaxis_title="Year",
        yaxis_title="Price (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(CHART_START, END_YEAR + 1)],
            ticktext=[str(y) for y in range(CHART_START, END_YEAR + 1)],
            gridcolor="#E0E0E0",
            range=[f"{CHART_START}-01", f"{END_YEAR}-12"],
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def price_percentile_paths(
    scenario_results: dict[str, ScenarioResult],
    scenario: str = "base",
) -> go.Figure:
    """
    Risk analysis chart: P5, P25, P50, P75, P95 percentile paths for one scenario.
    P95 area highlighted in red.
    """
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Scenario not available", **_LAYOUT_BASE)
        return fig

    df = result.monthly_prices.sort_values(["year", "month"])
    df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
    color = result.color

    fig = go.Figure()

    # P5–P95 extreme zone (red highlight)
    if "p5" in df.columns and "p95" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p95"], df["p5"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(183,28,28,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="P5–P95 (extreme range)",
            hoverinfo="skip",
        ))

    # P25–P75 zone
    if "p25" in df.columns and "p75" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p75"], df["p25"].iloc[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="P25–P75",
            hoverinfo="skip",
        ))

    # Percentile lines
    for pct, dash, width, label_suffix in [
        ("p95", "dot",   1.2, "P95 (worst 5%)"),
        ("p75", "dash",  1.0, "P75"),
        ("p50", "solid", 2.0, "P50 (median)"),
        ("p25", "dash",  1.0, "P25"),
        ("p5",  "dot",   1.2, "P5 (best 5%)"),
    ]:
        if pct not in df.columns:
            continue
        pct_color = "#B71C1C" if pct == "p95" else "#2E7D32" if pct == "p5" else color
        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df[pct],
            name=label_suffix,
            line=dict(color=pct_color, width=width, dash=dash),
            hovertemplate=f"%{{x}}: %{{y:.1f}} €/MWh<extra>{label_suffix}</extra>",
        ))

    sc_label = SCENARIO_LABELS.get(scenario, scenario)
    fig.update_layout(
        title=f"Price risk percentile paths – {sc_label}",
        xaxis_title="Month",
        yaxis_title="Price (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)],
            ticktext=[str(y) for y in range(START_YEAR, END_YEAR + 1)],
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MARKET DYNAMICS CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def tornado_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Tornado chart: price impact of variables €/MWh in base scenario for year 2030.

    sensitivity_df: DataFrame variable, impact_low, impact_high, value_low, value_high
    """
    fig = go.Figure()

    labels = sensitivity_df["variable"].tolist()

    # Negative impacts (price-reducing)
    neg_vals = [min(r["impact_low"], r["impact_high"]) for _, r in sensitivity_df.iterrows()]
    pos_vals = [max(r["impact_low"], r["impact_high"]) for _, r in sensitivity_df.iterrows()]

    fig.add_trace(go.Bar(
        y=labels,
        x=neg_vals,
        orientation="h",
        name="Price-reducing",
        marker_color="#2E7D32",
        hovertemplate="%{y}: %{x:.1f} €/MWh<extra>Reducing</extra>",
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=pos_vals,
        orientation="h",
        name="Price-increasing",
        marker_color="#B71C1C",
        hovertemplate="%{y}: %{x:.1f} €/MWh<extra>Increasing</extra>",
    ))

    fig.update_layout(
        title="Market variable price impact 2030 – base scenario",
        xaxis_title="Price impact (€/MWh)",
        barmode="overlay",
        height=420,
        xaxis=dict(gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#333", zerolinewidth=1.5),
        yaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT_BASE,
    )
    return fig


def datacenter_growth_chart(dc_df: pd.DataFrame) -> go.Figure:
    """Datacenter TWh growth curve annually."""
    colors = ["#B71C1C" if row["capped"] else "#1565C0" for _, row in dc_df.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dc_df["year"],
        y=dc_df["twh"],
        marker_color=colors,
        hovertemplate="%{x}: %{y:.2f} TWh<extra>Datacenters</extra>",
        name="Datacenters TWh",
    ))

    if dc_df["capped"].any():
        cap_year = dc_df[dc_df["capped"]]["year"].min()
        fig.add_vline(
            x=cap_year - 0.5,
            line_dash="dash",
            line_color="#B71C1C",
            annotation_text="50 TWh cap",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Datacenter electricity consumption in Finland 2025–2035",
        xaxis_title="Year",
        yaxis_title="Consumption (TWh/year)",
        height=360,
        yaxis=dict(gridcolor="#E0E0E0"),
        xaxis=dict(dtick=1, gridcolor="#E0E0E0"),
        showlegend=False,
        **_LAYOUT_BASE,
    )
    return fig


def interconnect_hintaero_chart(
    scenario_results: dict[str, ScenarioResult],
    max_hintaero: float,
) -> go.Figure:
    """
    Price difference FI vs estimated Nordic reference price monthly
    and maximum allowed price difference based on interconnection capacity.
    """
    fig = go.Figure()
    color_map = {"base": "#1565C0", "low": "#2E7D32", "high": "#B71C1C"}

    for scenario in ["low", "base", "high"]:
        result = scenario_results.get(scenario)
        if result is None:
            continue
        df = result.monthly_prices.sort_values(["year", "month"])
        df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        # Estimated Nord Pool reference price = FI price / (1 + hydro/other factors)
        # Simplification: reference ≈ FI P50 * 0.92
        df["nordpool_ref"] = df["p50"] * 0.92
        df["price_diff"] = df["p50"] - df["nordpool_ref"]
        color = color_map.get(scenario, "#555")
        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df["price_diff"],
            name=SCENARIO_LABELS.get(scenario, scenario),
            line=dict(color=color, width=1.5),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + SCENARIO_LABELS.get(scenario, scenario) + "</extra>",
        ))

    # Maximum allowed price difference based on interconnection capacity
    fig.add_hline(
        y=max_hintaero,
        line_dash="dash",
        line_color="#E65100",
        annotation_text=f"Max allowed price diff: {max_hintaero:.0f} €/MWh",
        annotation_position="top right",
    )

    fig.update_layout(
        title="FI–Nordic price difference and interconnection capacity limit",
        xaxis_title="Month",
        yaxis_title="Price difference (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)],
            ticktext=[str(y) for y in range(START_YEAR, END_YEAR + 1)],
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RISK ANALYSIS CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def hedge_comparison_chart(hedge_results: list, vol_mwh: float) -> go.Figure:
    """
    Bar chart: hedging strategy comparison P50 vs P95 costs.
    """
    from model.risk import HedgeResult

    labels = [h.strategy_name for h in hedge_results]
    cost_p50 = [h.annual_cost_p50 / 1000 for h in hedge_results]
    cost_p95 = [h.annual_cost_p95 / 1000 for h in hedge_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="P50 scenario (most likely)",
        x=labels,
        y=cost_p50,
        marker_color="#1565C0",
        hovertemplate="%{x}: %{y:.1f} k€<extra>P50</extra>",
    ))
    fig.add_trace(go.Bar(
        name="P95 scenario (worst 5%)",
        x=labels,
        y=cost_p95,
        marker_color="#B71C1C",
        hovertemplate="%{x}: %{y:.1f} k€<extra>P95</extra>",
    ))

    fig.update_layout(
        title=f"Hedging strategy cost comparison (volume {vol_mwh:,.0f} MWh/y)",
        xaxis_title="Strategy",
        yaxis_title="Annual cost (k€/y)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def efficient_frontier_chart(frontier_df: pd.DataFrame) -> go.Figure:
    """
    Efficient frontier chart: hedge cost vs CVaR 95%.
    x = additional cost vs spot P50 (€/y), y = CVaR 95% (€/y)
    """
    type_colors = {
        "Fixed":   "#1565C0",
        "Collar":  "#2E7D32",
        "Forward": "#F57C00",
    }

    fig = go.Figure()

    for type_name, grp in frontier_df.groupby("type"):
        color = type_colors.get(type_name, "#757575")
        fig.add_trace(go.Scatter(
            x=grp["hedge_cost"] / 1000,
            y=grp["cvar_95"] / 1000,
            mode="lines+markers",
            name=type_name,
            marker=dict(color=color, size=6),
            line=dict(color=color, width=1.5),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Additional cost: %{x:.0f} k€/y<br>"
                "CVaR 95%: %{y:.0f} k€/y<extra></extra>"
            ),
            text=grp["strategy"],
        ))

    # Mark spot point (0, spot_cvar)
    spot_row = frontier_df[frontier_df["strategy"] == "Fixed 0%"]
    if not spot_row.empty:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[float(spot_row["cvar_95"].iloc[0]) / 1000],
            mode="markers",
            marker=dict(color="#B71C1C", size=12, symbol="diamond"),
            name="Full spot",
            hovertemplate="Full spot<br>CVaR 95%: %{y:.0f} k€/y<extra></extra>",
        ))

    fig.update_layout(
        title="Efficient frontier – hedge cost vs risk (CVaR 95%)",
        xaxis_title="Additional cost vs spot P50 (k€/year)",
        yaxis_title="CVaR 95% (k€/year)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def stress_test_chart(stress_tests: list) -> go.Figure:
    """Bar chart: stress test price spike vs baseline."""
    from model.risk import StressTest

    names = [t.name for t in stress_tests]
    baseline = [t.baseline_price for t in stress_tests]
    spike = [t.price_spike for t in stress_tests]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Normal price (base P50)",
        x=names,
        y=baseline,
        marker_color="#1565C0",
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Normal</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Stress scenario",
        x=names,
        y=[s - b for s, b in zip(spike, baseline)],
        base=baseline,
        marker_color="#B71C1C",
        hovertemplate="%{x}: %{y:.1f} €/MWh additional<extra>Stress</extra>",
    ))

    fig.update_layout(
        title="Stress tests – price spike vs base scenario",
        xaxis_title="Stress scenario",
        yaxis_title="Price (€/MWh)",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def hedge_annual_cost_chart(
    hedge_df: pd.DataFrame,
    strategy_label: str,
    color: str = "#1565C0",
) -> go.Figure:
    """
    Line chart: effective price of selected hedge vs spot P50 annually.
    """
    fig = go.Figure()

    # Spot P50
    fig.add_trace(go.Scatter(
        x=hedge_df["year"],
        y=hedge_df["spot_p50"],
        name="Spot P50",
        line=dict(color="#757575", width=1.5, dash="dot"),
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Spot P50</extra>",
    ))

    # Hedged P10–P90 band
    fig.add_trace(go.Scatter(
        x=pd.concat([hedge_df["year"], hedge_df["year"].iloc[::-1]]),
        y=pd.concat([hedge_df["eff_price_p90"], hedge_df["eff_price_p10"].iloc[::-1]]),
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.15),
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Hedged P50
    fig.add_trace(go.Scatter(
        x=hedge_df["year"],
        y=hedge_df["eff_price_p50"],
        name=strategy_label,
        line=dict(color=color, width=2.5),
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + strategy_label + "</extra>",
    ))

    fig.update_layout(
        title=f"Effective price – {strategy_label} vs spot",
        xaxis_title="Year",
        yaxis_title="Price (€/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        xaxis=dict(dtick=1, gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY ANALYSIS CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def monthly_heatmap(scenario_results: dict[str, ScenarioResult], scenario: str = "base") -> go.Figure:
    """Heatmap: year × month, value = P50 price €/MWh."""
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Scenario not available", **_LAYOUT_BASE)
        return fig

    df = result.monthly_prices[["year", "month", "p50"]].copy()
    pivot = df.pivot(index="year", columns="month", values="p50")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[MONTH_LABELS_SHORT[m] for m in pivot.columns],
        y=pivot.index.astype(str).tolist(),
        colorscale="RdYlGn_r",
        hovertemplate="Year: %{y}, Month: %{x}<br>Price: %{z:.1f} €/MWh<extra></extra>",
        colorbar=dict(title="€/MWh"),
    ))
    fig.update_layout(
        title=f"Monthly price heatmap – {SCENARIO_LABELS.get(scenario, scenario)}",
        xaxis_title="Month",
        yaxis_title="Year",
        height=400,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def monthly_avg_bar(scenario_results: dict[str, ScenarioResult], scenario: str = "base") -> go.Figure:
    """Bar chart: monthly average prices (all years) for selected scenario."""
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Scenario not available", **_LAYOUT_BASE)
        return fig

    avg = result.monthly_prices.groupby("month")["p50"].mean().reset_index()
    avg["label"] = avg["month"].map(MONTH_LABELS_SHORT)

    bar_colors = [
        "#B71C1C" if m in [1, 2, 12] else
        "#2E7D32" if m in [6, 7, 8] else "#1565C0"
        for m in avg["month"]
    ]

    fig = go.Figure(go.Bar(
        x=avg["label"],
        y=avg["p50"],
        marker_color=bar_colors,
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"Monthly averages – {SCENARIO_LABELS.get(scenario, scenario)}",
        xaxis_title="Month",
        yaxis_title="Price (€/MWh)",
        height=360,
        yaxis=dict(gridcolor="#E0E0E0"),
        showlegend=False,
        **_LAYOUT_BASE,
    )
    return fig
