"""
report.py – Text summary and PDF report generation using the ReportLab library.
"""

import io
from datetime import date
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from model.scenarios import SCENARIO_LABELS, ScenarioResult, ScenarioParams


# ── Styles ────────────────────────────────────────────────────────────────────

def _h1() -> ParagraphStyle:
    return ParagraphStyle(
        "Header",
        parent=getSampleStyleSheet()["Heading1"],
        fontSize=18, textColor=colors.HexColor("#1B5E20"),
        spaceAfter=6, alignment=TA_CENTER,
    )


def _h2() -> ParagraphStyle:
    return ParagraphStyle(
        "SubHeader",
        parent=getSampleStyleSheet()["Heading2"],
        fontSize=13, textColor=colors.HexColor("#2E7D32"),
        spaceBefore=12, spaceAfter=4,
    )


def _body() -> ParagraphStyle:
    return ParagraphStyle(
        "Body",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
    )


# ── Summary text ─────────────────────────────────────────────────────────────

def generate_summary_text(
    scenario_results: dict[str, ScenarioResult],
    params: ScenarioParams,
    risk_summary: dict[str, Any] | None = None,
    data_notes: list[str] | None = None,
) -> str:
    """
    Generates an automatic text summary in English from market scenario
    and risk analysis results.
    """
    # Annual average P50 prices per scenario
    def avg_p50(scenario: str) -> float:
        r = scenario_results.get(scenario)
        if r is None:
            return 0.0
        return float(r.annual_prices["p50"].mean())

    low_avg  = avg_p50("low")
    base_avg = avg_p50("base")
    high_avg = avg_p50("high")

    # 2025 vs 2035 development
    def price_year(scenario: str, year: int) -> float:
        r = scenario_results.get(scenario)
        if r is None:
            return 0.0
        sub = r.annual_prices[r.annual_prices["year"] == year]
        return float(sub["p50"].values[0]) if not sub.empty else 0.0

    p_2025 = price_year("base", 2025)
    p_2035 = price_year("base", 2035)

    # Consumption growth
    total_growth = params.electrification_twh + params.ev_twh
    dc_final = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** 10)
    dc_final = min(dc_final, 50.0)

    text = (
        f"Analysis covers the years 2025–2035 with three scenarios. "
        f"In the base scenario, FI spot price averages {base_avg:.1f} €/MWh "
        f"(low {low_avg:.1f} €/MWh, high {high_avg:.1f} €/MWh). "
        f"From 2025 ({p_2025:.1f} €/MWh) to 2035 ({p_2035:.1f} €/MWh) "
        f"the price {'increases' if p_2035 > p_2025 else 'decreases'} in the base scenario "
        f"by {abs(p_2035 - p_2025):.1f} €/MWh. "
        f"Electrification and electric vehicles add {total_growth:.0f} TWh "
        f"of consumption by 2035. "
        f"Datacenter consumption is projected to grow "
        f"{params.datacenter_base_twh:.1f} → {dc_final:.1f} TWh. "
    )

    if risk_summary:
        rc = risk_summary.get("risk_class", "")
        cvar = risk_summary.get("cvar_95", 0.0)
        vol = risk_summary.get("volatility", 0.0)
        text += (
            f"Risk level is {rc} (volatility {vol:.1f} €/MWh, "
            f"CVaR 95% = {cvar:.1f} €/MWh). "
            f"{risk_summary.get('recommendation_text', '')} "
        )

    if data_notes:
        text += "Data assumptions: " + "; ".join(data_notes[:3]) + "."

    return text


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf_report(
    scenario_results: dict[str, ScenarioResult],
    params: ScenarioParams,
    n_simulations: int,
    r2: float = 0.0,
    risk_summary: dict[str, Any] | None = None,
    hedge_results: list | None = None,
    data_notes: list[str] | None = None,
) -> bytes:
    """
    Creates a PDF report from market analysis results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    story = []

    # Title
    story.append(Paragraph("Electricity Price Scenarios 2025–2035", _h1()))
    story.append(Paragraph(
        f"Finnish Electricity Market Analysis | {date.today().strftime('%d.%m.%Y')}",
        ParagraphStyle("sub", parent=_body(), alignment=TA_CENTER, textColor=colors.grey),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1B5E20"), spaceAfter=12))
    story.append(Spacer(1, 0.3 * cm))

    # Market parameters
    story.append(Paragraph("Market Parameters", _h2()))

    nuclear_map = {
        "current":       "Current level ≈ 4.4 GW",
        "ol3_hanhikivi": "OL3 + Hanhikivi replacement ≈ 5.4 GW",
        "new_plant":     "New plant ≈ 6 GW",
        "smr":           "SMR ≈ 4.9 GW",
    }
    hydro_map = {"normal": "Normal", "dry": "Dry year", "wet": "Wet year"}

    param_rows = [
        ["Parameter", "Value"],
        ["Wind power additional capacity", f"{params.wind_fi_gw:.1f} GW"],
        ["Solar energy growth", f"{params.solar_fi_gw:.1f} GW"],
        ["Nuclear FI", nuclear_map.get(params.nuclear_fi, params.nuclear_fi)],
        ["Nordic hydro", hydro_map.get(params.hydro_nordic, params.hydro_nordic)],
        ["Gas price", f"{params.gas_price_mwh:.0f} €/MWh"],
        ["CO₂ price", f"{params.co2_price_t:.0f} €/t"],
        ["Electrification + heat pumps", f"{params.electrification_twh:.0f} TWh"],
        ["Electric vehicles", f"{params.ev_twh:.1f} TWh"],
        ["Datacenters baseline", f"{params.datacenter_base_twh:.1f} TWh"],
        ["Datacenter growth rate", f"{params.datacenter_growth_pct:.0f} %/y"],
        ["Monte Carlo simulations", str(n_simulations)],
        ["Analysis period", "2025–2035"],
    ]
    if r2 > 0:
        param_rows.append(["Regression model R²", f"{r2:.3f}"])

    _add_table(story, param_rows,
               col_widths=[9 * cm, 7 * cm],
               header_color="#1B5E20",
               row_alt_color="#F1F8E9",
               grid_color="#C8E6C9")
    story.append(Spacer(1, 0.4 * cm))

    # Annual average prices per scenario
    story.append(Paragraph("Annual Average Prices 2025–2035 (P50, €/MWh)", _h2()))
    price_rows = [["Year", "Low", "Base", "High"]]
    years = list(range(2025, 2036))
    for year in years:
        row = [str(year)]
        for sc in ["low", "base", "high"]:
            r = scenario_results.get(sc)
            if r is not None:
                sub = r.annual_prices[r.annual_prices["year"] == year]
                val = f"{sub['p50'].values[0]:.1f}" if not sub.empty else "–"
            else:
                val = "–"
            row.append(val)
        price_rows.append(row)
    _add_table(story, price_rows,
               col_widths=[3 * cm, 4 * cm, 4 * cm, 4 * cm],
               header_color="#1565C0",
               row_alt_color="#E3F2FD",
               grid_color="#90CAF9",
               align_right_from=1)
    story.append(Spacer(1, 0.4 * cm))

    # Risk analysis
    if risk_summary:
        story.append(Paragraph("Risk Analysis", _h2()))
        story.append(Paragraph(risk_summary.get("recommendation_text", ""), _body()))
        risk_rows = [
            ["Metric", "Value"],
            ["Risk level", risk_summary.get("risk_class", "–").capitalize()],
            ["Volatility (€/MWh)", f"{risk_summary.get('volatility', 0):.1f}"],
            ["CVaR 95% (€/MWh)", f"{risk_summary.get('cvar_95', 0):.1f}"],
            ["Recommended strategy", risk_summary.get("best_strategy", "–")],
        ]
        _add_table(story, risk_rows,
                   col_widths=[9 * cm, 7 * cm],
                   header_color="#B71C1C",
                   row_alt_color="#FFEBEE",
                   grid_color="#FFCDD2")
        story.append(Spacer(1, 0.4 * cm))

    # Hedging strategy comparison
    if hedge_results:
        story.append(Paragraph("Hedging Strategy Comparison", _h2()))
        hedge_rows = [["Strategy", "Price P50\n(€/MWh)", "Price P95\n(€/MWh)", "Risk reduction\n(%)"]]
        for h in hedge_results:
            hedge_rows.append([
                h.strategy_name,
                f"{h.effective_price_p50:.1f}",
                f"{h.effective_price_p95:.1f}",
                f"{h.risk_reduction_ratio:.1f}%",
            ])
        _add_table(story, hedge_rows,
                   col_widths=[7 * cm, 3 * cm, 3 * cm, 3 * cm],
                   header_color="#37474F",
                   row_alt_color="#ECEFF1",
                   grid_color="#B0BEC5",
                   align_right_from=1)
        story.append(Spacer(1, 0.4 * cm))

    # Automatic summary
    story.append(Paragraph("Automatic Summary", _h2()))
    summary = generate_summary_text(scenario_results, params, risk_summary, data_notes)
    story.append(Paragraph(summary, _body()))
    story.append(Spacer(1, 0.5 * cm))

    # Data assumptions
    if data_notes:
        story.append(Paragraph("Data Assumptions Used", _h2()))
        for note in data_notes:
            story.append(Paragraph(f"• {note}", _body()))
        story.append(Spacer(1, 0.3 * cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6))
    story.append(Paragraph(
        "This report has been prepared based on Monte Carlo simulation and statistical models. "
        "Calculations are not guarantees of future prices. "
        "Use results as a directional tool for energy planning. "
        "Data is processed locally only — your data does not leave your machine.",
        ParagraphStyle("disc", parent=_body(), fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _add_table(
    story: list,
    rows: list,
    col_widths: list,
    header_color: str,
    row_alt_color: str,
    grid_color: str,
    align_right_from: int = None,
) -> None:
    """Helper function for adding a styled table."""
    t = Table(rows, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(row_alt_color)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(grid_color)),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    story.append(t)
