"""
Report Generator Module for AutoInsight AI
Assembles dataset insights, quality metrics, statistics, and visualizations
into a polished, multi-page corporate PDF report.
"""

import os
import uuid as uuid_lib
from datetime import datetime
from typing import Any, Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.logger import get_logger
from app.node_wrapper import node_error_boundary
from app.schemas.state import GraphState
from app.config import config

REPORTS_DIR = config.REPORTS_DIR
CHARTS_DIR = config.CHARTS_DIR

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# THEME — Centralized design system for corporate report styling
# ---------------------------------------------------------------------------
THEME = {
    "navy": colors.HexColor("#0F2540"),       # Main headings, cover page
    "gold": colors.HexColor("#C9A227"),       # Accent bars, highlights
    "teal": colors.HexColor("#1B6E7A"),       # Secondary accent
    "green": colors.HexColor("#2E8B57"),      # Growth/positive indicators
    "red": colors.HexColor("#C0392B"),        # Decline/negative indicators
    "grey_dark": colors.HexColor("#333333"),  # Body typography
    "grey_mid": colors.HexColor("#6B7280"),   # Captions and secondary labels
    "grey_light": colors.HexColor("#F4F5F7"), # Card & table container backgrounds
    "border": colors.HexColor("#E2E4E8"),     # Table & divider grid borders
}


class ReportGeneratorError(Exception):
    """Raised when the PDF report cannot be compiled or saved."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_text(text: Any) -> str:
    """Safely converts any input into a string to prevent rendering errors."""
    return str(text) if text is not None else ""


def _format_number(value: Any) -> str:
    """Formats raw numerical outputs into human-readable formatted strings."""
    if isinstance(value, float):
        if abs(value) < 10:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        elif value.is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.2f}"
    elif isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _add_page_footer(canvas: Any, doc: Any) -> None:
    """Draws a horizontal footer rule, page counter, and document branding on every page."""
    canvas.saveState()
    canvas.setStrokeColor(THEME["border"])
    canvas.line(0.6 * inch, 0.5 * inch, letter[0] - 0.6 * inch, 0.5 * inch)
    
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(THEME["grey_mid"])
    page_num_text = f"Page {doc.page}"
    canvas.drawCentredString(letter[0] / 2, 0.35 * inch, page_num_text)
    canvas.drawString(0.6 * inch, 0.35 * inch, "AutoInsight AI Report")
    canvas.restoreState()


def _section_divider() -> Table:
    """Generates a styled accent line section divider."""
    divider = Table([[""]], colWidths=[7.3 * inch], rowHeights=[0.03 * inch])
    divider.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), THEME["navy"]),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return divider


def _default_table_style() -> TableStyle:
    """Returns baseline styling for structured data tables."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), THEME["navy"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, THEME["border"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, THEME["grey_light"]]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])


def _build_styles() -> Any:
    """Defines custom paragraph styles for document typography."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=22, leading=26, fontName="Helvetica-Bold",
        textColor=THEME["navy"], spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", fontSize=11, leading=14, fontName="Helvetica",
        textColor=THEME["grey_mid"], spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=14, leading=18, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=8, textColor=THEME["navy"],
    ))
    styles.add(ParagraphStyle(
        name="SubHeading", fontSize=11, leading=15, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=4, textColor=THEME["navy"],
    ))
    styles.add(ParagraphStyle(
        name="Body", fontSize=10, leading=15, fontName="Helvetica",
        textColor=THEME["grey_dark"], spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader", fontSize=9, leading=11, fontName="Helvetica-Bold",
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell", fontSize=9, leading=12, fontName="Helvetica",
        textColor=THEME["grey_dark"],
    ))
    styles.add(ParagraphStyle(
        name="BulletBody", fontSize=10, leading=15, fontName="Helvetica",
        textColor=THEME["grey_dark"], leftIndent=12, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Small", fontSize=8, leading=11, textColor=THEME["grey_mid"],
    ))
    styles.add(ParagraphStyle(
        name="Caption", fontSize=8.5, leading=12, textColor=THEME["grey_mid"],
        alignment=1, spaceAfter=10, fontName="Helvetica-Oblique",
    ))
    return styles


# ---------------------------------------------------------------------------
# KPI Cards & Deterministic Computations Builder
# ---------------------------------------------------------------------------

def _compute_business_health_score(validation_report: dict, statistics: dict) -> int:
    """
    Deterministic 0-100 composite score. NOT an LLM guess — pure math
    from real numbers, so it's always accurate and reproducible.
    """
    val_data = validation_report or {}
    stats_data = statistics or {}
    score = val_data.get("health_score", 100)

    results = stats_data.get("results", {})
    outliers = results.get("outlier_detection", {})
    total_outliers = sum(v.get("outlier_count", 0) for v in outliers.values() if isinstance(v, dict))
    if total_outliers > 0:
        score -= min(total_outliers * 2, 15)  # cap penalty at 15 points

    return max(0, min(100, int(score)))


def _find_strongest_predictor(statistics: dict) -> str:
    """Finds the column with the highest absolute correlation to any other column."""
    stats_data = statistics or {}
    corr = stats_data.get("results", {}).get("correlation_analysis", {})
    if not corr or "skipped_reason" in corr:
        return "N/A"

    best_pair, best_val = None, 0
    for col_a, row in corr.items():
        if isinstance(row, dict):
            for col_b, val in row.items():
                if col_a != col_b and isinstance(val, (int, float)) and abs(val) > best_val:
                    best_val = abs(val)
                    best_pair = (col_a, col_b)

    return best_pair[0].replace("_", " ").title() if best_pair else "N/A"


def _extract_kpi_highlights(statistics: dict, validation_report: dict) -> list:
    """Extended KPI set: Business Health, Data Reliability, Strongest Predictor, plus existing cards."""
    kpis = []
    val_data = validation_report or {}

    try:
        health = _compute_business_health_score(val_data, statistics)
        kpis.append(("Business Health", f"{health}/100"))
    except Exception:
        pass

    try:
        reliability = val_data.get("health_score", 100)
        kpis.append(("Data Reliability", f"{reliability:.0f}%"))
    except Exception:
        pass

    try:
        predictor = _find_strongest_predictor(statistics)
        kpis.append(("Strongest Predictor", predictor))
    except Exception:
        pass

    try:
        results = (statistics or {}).get("results", {})
        outliers = results.get("outlier_detection", {})
        total_outliers = sum(v.get("outlier_count", 0) for v in outliers.values() if isinstance(v, dict))
        kpis.append(("Outliers Detected", str(total_outliers)))
    except Exception:
        pass

    return kpis[:4]


def _build_kpi_cards(kpis: List[Tuple[str, str]], styles: Any) -> List[Any]:
    """Renders extracted KPIs inside formatted summary cards."""
    if not kpis:
        return []
        
    elements = []
    row_size = min(4, len(kpis))
    if row_size == 0:
        return []
        
    for i in range(0, len(kpis), row_size):
        chunk = kpis[i:i + row_size]
        card_width = (6.3 / len(chunk)) * inch
        value_row = [Paragraph(f'<font size="16" color="#0F2540"><b>{v}</b></font>', styles["Body"]) for _, v in chunk]
        label_row = [Paragraph(f'<font size="8.5" color="#6B7280">{l}</font>', styles["Body"]) for l, _ in chunk]
        
        table = Table([value_row, label_row], colWidths=[card_width] * len(chunk))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), THEME["grey_light"]),
            ("LINEABOVE", (0, 0), (-1, 0), 2.5, THEME["gold"]),
            ("BOX", (0, 0), (-1, -1), 0.5, THEME["border"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, THEME["border"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.15 * inch))
        
    elements.append(Spacer(1, 0.15 * inch))
    return elements


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------

def _build_cover_page(profile: dict, styles) -> list:
    """Constructs a formal title cover page with robust metadata."""
    elements = []
    profile_data = profile or {}
    elements.append(Spacer(1, 1.6 * inch))

    bar = Table([[""]], colWidths=[1.4 * inch], rowHeights=[0.08 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), THEME["gold"])]))
    elements.append(bar)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("AutoInsight AI", ParagraphStyle(
            name="CoverTitle", fontSize=32, leading=38, fontName="Helvetica-Bold", textColor=THEME["navy"], spaceAfter=6)))
    elements.append(Paragraph("Automated Data Analysis Report", ParagraphStyle(
            name="CoverSubtitle", fontSize=14, leading=18, fontName="Helvetica", textColor=THEME["grey_mid"])))

    elements.append(Spacer(1, 0.5 * inch))
    timestamp = datetime.now().strftime("%B %d, %Y")
    report_id = f"AI-{uuid_lib.uuid4().hex[:8].upper()}"

    meta_lines = [
        f"Report ID: {report_id}",
        f"Prepared: {timestamp}",
        f"Dataset: {profile_data.get('row_count', 'N/A')} rows &middot; {profile_data.get('column_count', 'N/A')} columns",
        "AutoInsight AI Engine v1.0",
        "CONFIDENTIAL — Prepared for internal business use only.",
    ]
    for line in meta_lines:
        elements.append(Paragraph(line, styles["Small"]))

    elements.append(PageBreak())
    return elements


def _build_title_section(styles: Any) -> List[Any]:
    """Renders the executive report header on subsequent pages."""
    elements = []
    accent = Table([[""]], colWidths=[1.2 * inch], rowHeights=[0.06 * inch])
    accent.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), THEME["teal"]),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(accent)
    elements.append(Spacer(1, 0.12 * inch))
    
    elements.append(Paragraph("AutoInsight AI", styles["ReportTitle"]))
    elements.append(Paragraph("Automated Data Analysis Report", styles["Subtitle"]))
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    elements.append(Paragraph(f"Generated on {timestamp}", styles["Small"]))
    elements.append(Spacer(1, 0.25 * inch))
    return elements


def _build_executive_summary(insights: List[str], plan: dict, styles: Any) -> List[Any]:
    """Builds the Executive Summary section."""
    elements = []
    elements.append(Paragraph("Execution Plan Summary", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    plan_summary = plan.get("summary", "") if isinstance(plan, dict) else ""
    if plan_summary:
        elements.append(Paragraph(plan_summary, styles["Body"]))
        elements.append(Spacer(1, 0.12 * inch))

    if insights:
        elements.append(Paragraph("Core Insights", styles["SubHeading"]))
        rows = [[Paragraph(f"<b>{i+1}.</b> {text}", styles["BulletBody"])] for i, text in enumerate(insights)]
        insight_table = Table(rows, colWidths=[6.3 * inch])
        insight_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), THEME["grey_light"]),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, THEME["border"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(insight_table)
    else:
        elements.append(Paragraph("No specific insights generated for this dataset.", styles["Body"]))

    elements.append(Spacer(1, 0.25 * inch))
    return elements


def _build_executive_snapshot(profile: dict, plan: dict, statistics: dict, validation_report: dict, styles: Any) -> List[Any]:
    """Builds a compact boardroom-style snapshot with the most important dataset facts."""
    results = statistics.get("results", {}) if isinstance(statistics, dict) else {}
    profile_data = profile or {}

    snapshot_rows = [[
        Paragraph("Dimension", styles["TableHeader"]),
        Paragraph("Value", styles["TableHeader"]),
        Paragraph("Why it matters", styles["TableHeader"]),
    ]]

    snapshot_rows.append([
        Paragraph("Dataset size", styles["TableCell"]),
        Paragraph(f"{profile_data.get('row_count', 'N/A')} rows x {profile_data.get('column_count', 'N/A')} columns", styles["TableCell"]),
        Paragraph("Defines the scope and confidence of the analysis.", styles["TableCell"]),
    ])

    if validation_report:
        health_score = validation_report.get("health_score")
        if health_score is not None:
            snapshot_rows.append([
                Paragraph("Data health", styles["TableCell"]),
                Paragraph(f"{health_score} / 100", styles["TableCell"]),
                Paragraph("Confirms the quality of the source data before interpretation.", styles["TableCell"]),
            ])

    trend = results.get("trend_analysis", {}) if isinstance(results, dict) else {}
    if isinstance(trend, dict) and trend and "skipped_reason" not in trend:
        first_metric_name = next(iter(trend.keys()), None)
        first_metric = trend.get(first_metric_name, {}) if first_metric_name else {}
        if isinstance(first_metric, dict) and first_metric.get("percent_change") is not None:
            snapshot_rows.append([
                Paragraph("Primary trend", styles["TableCell"]),
                Paragraph(f"{first_metric_name.replace('_', ' ').title()} {first_metric['percent_change']:.1f}%", styles["TableCell"]),
                Paragraph("Shows the main direction of movement across the observed period.", styles["TableCell"]),
            ])

    if plan and plan.get("summary"):
        snapshot_rows.append([
            Paragraph("Plan summary", styles["TableCell"]),
            Paragraph("Execution plan confirmed", styles["TableCell"]),
            Paragraph(plan.get("summary", ""), styles["TableCell"]),
        ])

    table = Table(snapshot_rows, colWidths=[1.4 * inch, 2.15 * inch, 3.15 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), THEME["navy"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, THEME["border"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, THEME["grey_light"]]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements = []
    elements.append(Paragraph("Executive Snapshot", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(table)
    elements.append(Spacer(1, 0.22 * inch))
    return elements


def _build_key_findings_section(statistics: dict, insights: List[str], validation_report: dict, styles: Any) -> List[Any]:
    """Builds a concise key findings section from the strongest available signals."""
    findings = []
    results = statistics.get("results", {}) if isinstance(statistics, dict) else {}

    trend = results.get("trend_analysis", {}) if isinstance(results, dict) else {}
    if isinstance(trend, dict) and trend and "skipped_reason" not in trend:
        best_metric = None
        best_change = None
        for metric, data in trend.items():
            if isinstance(data, dict) and data.get("percent_change") is not None:
                change = float(data["percent_change"])
                if best_change is None or abs(change) > abs(best_change):
                    best_metric = metric
                    best_change = change
        if best_metric is not None and best_change is not None:
            direction = "increased" if best_change >= 0 else "declined"
            findings.append(
                f"<b>{best_metric.replace('_', ' ').title()}</b> {direction} by {abs(best_change):.1f}% over the measured period."
            )

    corr = results.get("correlation_analysis", {}) if isinstance(results, dict) else {}
    if isinstance(corr, dict) and corr and "skipped_reason" not in corr:
        cols = list(corr.keys())
        strongest_pair = None
        strongest_value = None
        for left in cols:
            row = corr.get(left, {})
            if not isinstance(row, dict):
                continue
            for right, value in row.items():
                if left == right or not isinstance(value, (int, float)):
                    continue
                if strongest_value is None or abs(value) > abs(strongest_value):
                    strongest_pair = (left, right)
                    strongest_value = float(value)
        if strongest_pair and strongest_value is not None:
            findings.append(
                f"<b>{strongest_pair[0].replace('_', ' ').title()}</b> and <b>{strongest_pair[1].replace('_', ' ').title()}</b> show a correlation of {strongest_value:.2f}."
            )

    outliers = results.get("outlier_detection", {}) if isinstance(results, dict) else {}
    if isinstance(outliers, dict):
        total_outliers = sum(v.get("outlier_count", 0) for v in outliers.values() if isinstance(v, dict))
        if total_outliers:
            findings.append(f"Detected <b>{total_outliers}</b> outlier record(s) across the numeric measures.")

    health_score = validation_report.get("health_score") if isinstance(validation_report, dict) else None
    if isinstance(health_score, (int, float)):
        if health_score >= 95:
            findings.append(f"Source data quality is strong at <b>{health_score} / 100</b>, so the analysis has a reliable foundation.")
        elif health_score >= 85:
            findings.append(f"Source data quality is acceptable at <b>{health_score} / 100</b>, but targeted cleanup opportunities remain.")
        else:
            findings.append(f"Source data quality is limited at <b>{health_score} / 100</b> and should be reviewed before major decisions.")

    if insights and len(insights) > 0:
        findings.append(f"Top generated insight: {insights[0]}")

    if not findings:
        findings.append("No strong analytical anomalies were detected; the dataset appears relatively stable.")

    elements = []
    elements.append(Paragraph("Key Findings", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    rows = [[Paragraph(f"→ {finding}", styles["BulletBody"])] for finding in findings[:4]]
    table = Table(rows, colWidths=[6.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), THEME["grey_light"]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, THEME["border"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.22 * inch))
    return elements


def _build_dataset_overview(profile: dict, styles: Any) -> List[Any]:
    """Formats dataset structural dimensions and column classifications."""
    elements = []
    elements.append(Paragraph("Dataset Overview", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    profile_data = profile or {}
    overview_data = [
        [Paragraph("Metric", styles["TableHeader"]), Paragraph("Value", styles["TableHeader"])],
        [Paragraph("Total Rows", styles["TableCell"]), Paragraph(str(profile_data.get("row_count", "N/A")), styles["TableCell"])],
        [Paragraph("Total Columns", styles["TableCell"]), Paragraph(str(profile_data.get("column_count", "N/A")), styles["TableCell"])],
        [Paragraph("Numeric Columns", styles["TableCell"]), Paragraph(str(len(profile_data.get("numeric_columns", []))), styles["TableCell"])],
        [Paragraph("Categorical Columns", styles["TableCell"]), Paragraph(str(len(profile_data.get("categorical_columns", []))), styles["TableCell"])],
        [Paragraph("Datetime Columns", styles["TableCell"]), Paragraph(str(len(profile_data.get("datetime_columns", []))), styles["TableCell"])],
    ]

    table = Table(overview_data, colWidths=[3.3 * inch, 2.3 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))
    return elements


def _build_data_reliability_checklist(validation_report: dict, cleaning_report: dict, styles) -> list:
    """Deterministic pass/warn/fail checklist — every check computed from real numbers."""
    elements = []
    elements.append(Paragraph("Data Reliability Checklist", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    val_data = validation_report or {}
    clean_data = cleaning_report or {}

    missing_pct = val_data.get("missing_percent", 0)
    dup_count = val_data.get("duplicate_rows", 0)
    health = val_data.get("health_score", 100)

    checks = [
        ("Missing Values", "PASS" if missing_pct < 5 else ("WARN" if missing_pct < 15 else "FAIL"),
         f"{missing_pct:.1f}% missing"),
        ("Duplicate Rows", "PASS" if dup_count == 0 else ("WARN" if dup_count < 10 else "FAIL"),
         f"{dup_count} duplicate(s) found"),
        ("Overall Data Health", "PASS" if health >= 85 else ("WARN" if health >= 60 else "FAIL"),
         f"{health}/100"),
        ("Cleaning Applied", "PASS", f"{clean_data.get('duplicates_removed', 0)} row(s) cleaned"),
    ]

    status_colors = {"PASS": THEME["green"], "WARN": THEME["gold"], "FAIL": THEME["red"]}
    rows = [[
        Paragraph("Check", styles["TableHeader"]),
        Paragraph("Status", styles["TableHeader"]),
        Paragraph("Detail", styles["TableHeader"])
    ]]
    for name, status, detail in checks:
        color = status_colors[status]
        status_cell = Paragraph(f'<font color="{color}"><b>&#9679; {status}</b></font>', styles["TableCell"])
        rows.append([
            Paragraph(name, styles["TableCell"]),
            status_cell,
            Paragraph(_safe_text(detail), styles["TableCell"])
        ])

    table = Table(rows, colWidths=[2.2 * inch, 1.2 * inch, 2.9 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_business_story_section(narrative: dict, styles) -> list:
    elements = []
    elements.append(Paragraph("Business Story", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    narrative_data = narrative or {}
    story_text = narrative_data.get("business_story", "")
    elements.append(Paragraph(_safe_text(story_text), styles["Body"]))
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_column_intelligence_section(narrative: dict, styles) -> list:
    elements = []
    elements.append(Paragraph("Column Intelligence", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    narrative_data = narrative or {}
    for col_info in narrative_data.get("column_intelligence", []):
        if not isinstance(col_info, dict):
            continue
        elements.append(Paragraph(_safe_text(col_info.get("column", "")), styles["SubHeading"]))
        rows = [
            [Paragraph("<b>Purpose</b>", styles["TableCell"]), Paragraph(_safe_text(col_info.get("purpose", "")), styles["TableCell"])],
            [Paragraph("<b>Business Interpretation</b>", styles["TableCell"]), Paragraph(_safe_text(col_info.get("business_interpretation", "")), styles["TableCell"])],
            [Paragraph("<b>Risk Note</b>", styles["TableCell"]), Paragraph(_safe_text(col_info.get("risk_note", "")), styles["TableCell"])],
        ]
        table = Table(rows, colWidths=[1.8 * inch, 4.5 * inch])
        table.setStyle(_default_table_style())
        elements.append(table)
        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Spacer(1, 0.15 * inch))
    return elements


def _stats_dict_to_table_rows(data: dict, styles: Any) -> List[List[Any]]:
    """Converts nested statistics dictionaries into table rows."""
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str)):
                    if sub_key == "percent_change" and isinstance(sub_value, (int, float)):
                        color = THEME["green"].hexval() if sub_value >= 0 else THEME["red"].hexval()
                        arrow = "▲" if sub_value >= 0 else "▼"
                        display_val = f'<font color="{color}"><b>{arrow} {abs(sub_value):.1f}%</b></font>'
                    else:
                        display_val = _format_number(sub_value)
                    
                    metric_label = f"{key.replace('_', ' ').title()} ({sub_key.replace('_', ' ')})"
                    rows.append([
                        Paragraph(metric_label, styles["TableCell"]),
                        Paragraph(display_val, styles["TableCell"])
                    ])
        elif isinstance(value, (int, float, str)):
            rows.append([
                Paragraph(key.replace('_', ' ').title(), styles["TableCell"]),
                Paragraph(_format_number(value), styles["TableCell"])
            ])
    return rows


def _build_statistics_section(statistics: dict, styles: Any) -> List[Any]:
    """Builds the Statistical Summary section."""
    elements = []
    elements.append(Paragraph("Statistical Analyses", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    results = (statistics or {}).get("results", {}) if isinstance(statistics, dict) else {}
    if not results:
        elements.append(Paragraph("No statistical analyses were completed.", styles["Body"]))
        return elements

    for analysis_name, analysis_result in results.items():
        elements.append(Paragraph(analysis_name.replace("_", " ").title(), styles["SubHeading"]))

        if isinstance(analysis_result, dict) and "skipped_reason" in analysis_result:
            elements.append(Paragraph(f"Skipped — {analysis_result['skipped_reason']}", styles["Small"]))
        elif isinstance(analysis_result, dict) and "error" in analysis_result:
            elements.append(Paragraph(f"Error — {analysis_result['error']}", styles["Small"]))
        
        elif isinstance(analysis_result, dict) and "groups" in analysis_result:
            group_col = analysis_result.get("grouped_by", "Group")
            measure_col = analysis_result.get("measured", "Value")
            elements.append(Paragraph(f"Grouped by: <b>{group_col.title()}</b> | Measure: <b>{measure_col.title()}</b>", styles["Small"]))
            elements.append(Spacer(1, 0.05 * inch))
            
            rows = [[
                Paragraph("Group", styles["TableHeader"]),
                Paragraph("Total", styles["TableHeader"]),
                Paragraph("Average", styles["TableHeader"]),
                Paragraph("Count", styles["TableHeader"])
            ]]
            for group_name, stats in analysis_result["groups"].items():
                rows.append([
                    Paragraph(str(group_name), styles["TableCell"]),
                    Paragraph(_format_number(stats.get("total", 0)), styles["TableCell"]),
                    Paragraph(_format_number(stats.get("average", 0)), styles["TableCell"]),
                    Paragraph(str(stats.get("count", 0)), styles["TableCell"])
                ])
            
            table = Table(rows, colWidths=[2 * inch, 1.4 * inch, 1.4 * inch, 1 * inch])
            table.setStyle(_default_table_style())
            elements.append(table)
            
        elif isinstance(analysis_result, dict):
            rows = _stats_dict_to_table_rows(analysis_result, styles)
            if rows:
                table_data = [[Paragraph("Metric", styles["TableHeader"]), Paragraph("Value", styles["TableHeader"])]] + rows
                table = Table(table_data, colWidths=[3.3 * inch, 2.3 * inch])
                table.setStyle(_default_table_style())
                elements.append(table)
            else:
                elements.append(Paragraph("No displayable metric entries.", styles["Small"]))
        else:
            elements.append(Paragraph(str(analysis_result), styles["Body"]))

        elements.append(Spacer(1, 0.18 * inch))

    return elements


def _build_relationship_analysis_section(statistics: dict, styles) -> list:
    elements = []
    elements.append(Paragraph("Relationship Analysis", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    corr = (statistics or {}).get("results", {}).get("correlation_analysis", {})
    if not corr or "skipped_reason" in corr:
        elements.append(Paragraph("Not enough numeric columns for relationship analysis.", styles["Body"]))
        return elements

    seen_pairs = set()
    relationships = []
    for col_a, row in corr.items():
        if isinstance(row, dict):
            for col_b, val in row.items():
                if col_a == col_b or not isinstance(val, (int, float)):
                    continue
                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
    
                abs_val = abs(val)
                if abs_val >= 0.7:
                    strength, stars = "Strong", "&#9733;" * 5
                elif abs_val >= 0.4:
                    strength, stars = "Medium", "&#9733;" * 3
                else:
                    strength, stars = "Weak", "&#9733;"
    
                relationships.append((abs_val, col_a, col_b, val, strength, stars))

    relationships.sort(reverse=True)
    if not relationships:
        elements.append(Paragraph("No calculable correlation relationships observed.", styles["Body"]))
        return elements

    rows = [[
        Paragraph("<b>Relationship</b>", styles["TableHeader"]), 
        Paragraph("<b>Correlation</b>", styles["TableHeader"]), 
        Paragraph("<b>Strength</b>", styles["TableHeader"])
    ]]
    for _, col_a, col_b, val, strength, stars in relationships[:8]:
        pair_name = f"{col_a.replace('_',' ').title()} ↔ {col_b.replace('_',' ').title()}"
        rows.append([
            Paragraph(pair_name, styles["TableCell"]), 
            Paragraph(f"{val:.3f}", styles["TableCell"]), 
            Paragraph(f'{strength} <font color="{THEME["gold"]}">{stars}</font>', styles["TableCell"])
        ])

    table = Table(rows, colWidths=[3 * inch, 1.3 * inch, 2.2 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_driver_ranking_section(statistics: dict, styles) -> list:
    """Ranks each numeric column by how strongly it relates to everything else — a proxy for 'importance'."""
    elements = []
    elements.append(Paragraph("Driver Ranking", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    corr = (statistics or {}).get("results", {}).get("correlation_analysis", {})
    if not corr or "skipped_reason" in corr:
        elements.append(Paragraph("Not enough numeric columns to rank drivers.", styles["Body"]))
        return elements

    avg_strength = {}
    for col, row in corr.items():
        if isinstance(row, dict):
            others = [abs(v) for k, v in row.items() if k != col and isinstance(v, (int, float))]
            if others:
                avg_strength[col] = sum(others) / len(others)

    ranked = sorted(avg_strength.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        elements.append(Paragraph("Could not compute reliable driver rankings.", styles["Body"]))
        return elements

    rows = [[
        Paragraph("Rank", styles["TableHeader"]),
        Paragraph("Variable", styles["TableHeader"]),
        Paragraph("Average Relationship Strength", styles["TableHeader"])
    ]]
    for i, (col, strength) in enumerate(ranked[:6], 1):
        stars = "&#9733;" * max(1, round(strength * 5))
        rows.append([
            Paragraph(str(i), styles["TableCell"]),
            Paragraph(col.replace("_", " ").title(), styles["TableCell"]),
            Paragraph(f'<font color="{THEME["gold"]}">{stars}</font>', styles["TableCell"])
        ])

    table = Table(rows, colWidths=[0.7 * inch, 2.8 * inch, 3 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_opportunities_risks_section(narrative: dict, statistics: dict, validation_report: dict, styles) -> list:
    elements = []
    narrative_data = narrative or {}
    val_data = validation_report or {}
    stats_data = statistics or {}

    # Opportunities (from narrative agent)
    elements.append(Paragraph("Opportunities", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    for i, opp in enumerate(narrative_data.get("opportunities", []), 1):
        if not isinstance(opp, dict):
            continue
        elements.append(Paragraph(
            f"<b>{i}. {_safe_text(opp.get('title',''))}</b> (Confidence: {_safe_text(opp.get('confidence','Medium'))})",
            styles["SubHeading"]))
        elements.append(Paragraph(_safe_text(opp.get("recommendation", "")), styles["Body"]))
        elements.append(Paragraph(f"<i>Expected impact: {_safe_text(opp.get('expected_impact',''))}</i>", styles["Small"]))
        elements.append(Spacer(1, 0.1 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # Risks (from narrative agent)
    elements.append(Paragraph("Risks", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    priority_colors = {"Critical": THEME["red"], "High": THEME["red"], "Medium": THEME["gold"], "Low": THEME["grey_mid"]}
    for risk in narrative_data.get("risks", []):
        if not isinstance(risk, dict):
            continue
        color = priority_colors.get(risk.get("priority", "Medium"), THEME["gold"])
        elements.append(Paragraph(
            f'<font color="{color}"><b>[{_safe_text(risk.get("priority","Medium"))}]</b></font> '
            f'<b>{_safe_text(risk.get("title",""))}</b> — {_safe_text(risk.get("description",""))}',
            styles["Body"]))
        elements.append(Spacer(1, 0.08 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # AI Alerts (deterministic)
    elements.append(Paragraph("Alerts", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    alerts = []
    health = val_data.get("health_score", 100)
    if health == 100 and val_data.get("missing_cells", 0) == 0:
        alerts.append(("POSITIVE", THEME["green"], "Dataset quality is excellent — no missing values detected."))
    if health < 70:
        alerts.append(("CRITICAL", THEME["red"], f"Data health score is low ({health}/100) — investigate data collection process."))
    outliers = stats_data.get("results", {}).get("outlier_detection", {})
    total_outliers = sum(v.get("outlier_count", 0) for v in outliers.values() if isinstance(v, dict))
    if total_outliers > 0:
        alerts.append(("WARNING", THEME["gold"], f"{total_outliers} outlier record(s) detected across numeric fields."))

    if not alerts:
        alerts.append(("INFO", THEME["teal"], "No major alerts or critical system errors detected."))

    for label, color, text in alerts:
        elements.append(Paragraph(f'<font color="{color}"><b>[{label}]</b></font> {text}', styles["Body"]))
        elements.append(Spacer(1, 0.06 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _generate_recommendations(statistics: dict, validation_report: dict) -> List[str]:
    """Derives deterministic recommendations from statistical findings and health checks."""
    recs = []
    results = (statistics or {}).get("results", {}) if isinstance(statistics, dict) else {}
    trend = results.get("trend_analysis", {}) if isinstance(results, dict) else {}
    
    if isinstance(trend, dict):
        for metric, data in trend.items():
            if isinstance(data, dict) and data.get("percent_change") is not None:
                change = data["percent_change"]
                if change < 0:
                    recs.append(f"<b>{metric.replace('_', ' ').title()}</b> declined by {abs(change):.1f}% over the evaluated timeframe. Investigate drivers.")
                elif change > 20:
                    recs.append(f"<b>{metric.replace('_', ' ').title()}</b> grew by {change:.1f}%. Capitalize on key drivers behind this growth.")
                
    outliers = results.get("outlier_detection", {}) if isinstance(results, dict) else {}
    if isinstance(outliers, dict):
        for metric, data in outliers.items():
            if isinstance(data, dict) and data.get("outlier_count", 0) > 0:
                recs.append(f"Detected {data['outlier_count']} outlier record(s) in <b>{metric.replace('_', ' ')}</b>. Review records for data entry anomalies.")
            
    corr = results.get("correlation_analysis", {}) if isinstance(results, dict) else {}
    if isinstance(corr, dict):
        cols = list(corr.keys())
        if len(cols) >= 2 and isinstance(corr.get(cols[0]), dict):
            val = corr[cols[0]].get(cols[1])
            if val is not None and abs(val) > 0.7:
                recs.append(f"Strong correlation ({val:.2f}) observed between <b>{cols[0].replace('_',' ')}</b> and <b>{cols[1].replace('_',' ')}</b>.")
            
    health_score = (validation_report or {}).get("health_score", 100) if isinstance(validation_report, dict) else 100
    if isinstance(health_score, (int, float)) and health_score < 90:
        recs.append("Data hygiene opportunities detected. Establish automated collection validation to resolve missing values.")
        
    if not recs:
        recs.append("Data indicators remain within baseline bounds. Maintain standard operational monitoring.")
        
    return recs[:5]


def _build_recommendations_section(statistics: dict, validation_report: dict, styles: Any) -> List[Any]:
    """Builds the Actionable Recommendations section."""
    elements = []
    elements.append(Paragraph("Actionable Recommendations", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    
    recs = _generate_recommendations(statistics, validation_report)
    rows = [[Paragraph(f"→ {rec}", styles["BulletBody"])] for rec in recs]
    
    table = Table(rows, colWidths=[6.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), THEME["grey_light"]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, THEME["border"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))
    return elements


def _build_charts_section(visualizations: dict, styles: Any) -> List[Any]:
    """Renders charts and figures into the PDF layout."""
    elements = []
    elements.append(Paragraph("Visualizations & Charts", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.15 * inch))

    generated = (visualizations or {}).get("generated", []) if isinstance(visualizations, dict) else []
    if not generated:
        elements.append(Paragraph("No chart figures were generated for this dataset.", styles["Body"]))
        return elements

    for chart in generated:
        if not isinstance(chart, dict):
            continue
        file_path = chart.get("file_path")
        if not file_path or not os.path.exists(file_path):
            elements.append(Paragraph(f"[Chart missing: {chart.get('chart_type')}]", styles["Body"]))
            continue

        chart_block = []
        chart_title = str(chart.get("chart_type", "")).replace("_", " ").title() + " Chart"
        chart_block.append(Paragraph(chart_title, styles["SubHeading"]))

        img = Image(file_path, width=5.8 * inch, height=3.3 * inch)
        chart_block.append(img)
        if chart.get("reason"):
            chart_block.append(Paragraph(str(chart.get("reason")), styles["Caption"]))
        chart_block.append(Spacer(1, 0.2 * inch))

        # Keeps title, image, and caption together on the same page
        elements.append(KeepTogether(chart_block))

    return elements


def _build_executive_conclusion(narrative: dict, statistics: dict, validation_report: dict, styles) -> list:
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("Executive Conclusion", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    narrative_data = narrative or {}
    health = _compute_business_health_score(validation_report, statistics)
    status = "Healthy" if health >= 80 else ("Needs Attention" if health >= 60 else "At Risk")
    top_driver = _find_strongest_predictor(statistics)
    
    risks = narrative_data.get("risks", [])
    top_risk = risks[0].get("title", "None identified") if risks and isinstance(risks[0], dict) else "None identified"

    rows = [
        [Paragraph("<b>Business Status</b>", styles["TableCell"]), Paragraph(status, styles["TableCell"])],
        [Paragraph("<b>Primary Driver</b>", styles["TableCell"]), Paragraph(top_driver, styles["TableCell"])],
        [Paragraph("<b>Top Risk</b>", styles["TableCell"]), Paragraph(top_risk, styles["TableCell"])],
        [Paragraph("<b>Overall Health Score</b>", styles["TableCell"]), Paragraph(f"{health}/100", styles["TableCell"])],
    ]
    table = Table(rows, colWidths=[2.5 * inch, 3.8 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_appendix(styles) -> list:
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("Appendix", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(
        "<b>AI Limitations:</b> This report is generated from historical data only. It does not include "
        "forecasts, predictions, or causal claims — correlation does not imply causation. Recommendations "
        "should be validated against business context before acting.",
        styles["Small"]))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(
        "<b>Methodology:</b> Statistical analyses (descriptive statistics, correlation, outlier detection via "
        "IQR method) were computed directly from the cleaned dataset. Narrative sections were generated by an "
        "LLM constrained to reference only the computed statistics provided to it.",
        styles["Small"]))
    return elements


# ---------------------------------------------------------------------------
# LangGraph Node Implementation
# ---------------------------------------------------------------------------

@node_error_boundary("report_generator")
def report_generator_node(state: GraphState) -> GraphState:
    """
    LangGraph pipeline node for deterministic PDF generation.
    
    Assembles profile, validation, statistical analysis, and visual image paths
    into a structured PDF report saved to disk.
    """
    logger.info("Report Generator Node started")

    profile = state.get("profile") or {}
    validation_report = state.get("validation_report")
    cleaning_report = state.get("cleaning_report")
    statistics = state.get("statistics")
    raw_insights = state.get("insights")
    narrative = state.get("narrative") or {}

    missing = [name for name, val in {
        "validation_report": validation_report,
        "cleaning_report": cleaning_report,
        "statistics": statistics,
        "insights": raw_insights,
    }.items() if val is None]

    if missing:
        raise ReportGeneratorError(
            f"Report Generator is missing required state fields: {missing}. "
            "Check that every upstream node wrote to its canonical GraphState key."
        )

    plan = state.get("plan") or {}

    charts_state = state.get("visualizations", {})
    if isinstance(charts_state, dict):
        visualizations = charts_state
    elif isinstance(charts_state, list):
        visualizations = {"generated": charts_state}
    else:
        visualizations = {"generated": []}

    insights = raw_insights if isinstance(raw_insights, list) else [str(raw_insights)]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    file_name = f"report_{uuid_lib.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(REPORTS_DIR, file_name)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        report_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    story = []
    story += _build_cover_page(profile, styles)
    story += _build_title_section(styles)

    kpis = _extract_kpi_highlights(statistics, validation_report)
    story += _build_kpi_cards(kpis, styles)
    
    story += _build_executive_snapshot(profile, plan, statistics, validation_report, styles)
    story += _build_key_findings_section(statistics, insights, validation_report, styles)

    if narrative:
        story += _build_business_story_section(narrative, styles)

    story += _build_executive_summary(insights, plan, styles)
    story += _build_dataset_overview(profile, styles)
    story += _build_data_reliability_checklist(validation_report, cleaning_report, styles)

    if narrative:
        story += _build_column_intelligence_section(narrative, styles)

    story += _build_statistics_section(statistics, styles)
    story += _build_relationship_analysis_section(statistics, styles)
    story += _build_driver_ranking_section(statistics, styles)

    if narrative:
        story += _build_opportunities_risks_section(narrative, statistics, validation_report, styles)

    story += _build_recommendations_section(statistics, validation_report, styles)
    story += _build_charts_section(visualizations, styles)

    if narrative:
        story += _build_executive_conclusion(narrative, statistics, validation_report, styles)

    story += _build_appendix(styles)

    try:
        doc.build(story, onFirstPage=_add_page_footer, onLaterPages=_add_page_footer)
        logger.info(f"Report successfully compiled and saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to compile PDF report: {e}")
        raise ReportGeneratorError(f"Failed to build PDF report: {e}")

    return {**state, "report_path": report_path}