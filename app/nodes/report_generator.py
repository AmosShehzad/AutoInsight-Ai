"""
Report Generator Module for AutoInsight AI
Assembles dataset insights, quality metrics, statistics, and visualizations
into a polished, multi-page corporate PDF report.
"""

import os
import uuid
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
from app.schemas.state import GraphState

logger = get_logger(__name__)

REPORTS_DIR = "outputs/reports"

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
# Formatting & Page Mechanics
# ---------------------------------------------------------------------------

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
# KPI Cards Builder
# ---------------------------------------------------------------------------

def _extract_kpi_highlights(statistics: dict) -> List[Tuple[str, str]]:
    """Extracts up to 4 key performance indicators from statistical results."""
    kpis = []
    results = statistics.get("results", {}) if isinstance(statistics, dict) else {}

    # 1. Trend Analysis KPI
    try:
        trend = results.get("trend_analysis", {})
        if trend and "skipped_reason" not in trend:
            first_metric = next(iter(trend.values()), None)
            if first_metric and first_metric.get("percent_change") is not None:
                metric_name = next(iter(trend.keys()))
                change = first_metric["percent_change"]
                direction = "▲" if change >= 0 else "▼"
                kpis.append((f"{metric_name.replace('_', ' ').title()} Growth", f"{direction} {abs(change):.1f}%"))
    except Exception as e:
        logger.debug(f"KPI trend extraction skipped: {e}")

    # 2. Descriptive Stats KPI
    try:
        desc = results.get("descriptive_stats", {})
        if desc and "skipped_reason" not in desc:
            first_col = next(iter(desc.keys()), None)
            if first_col and desc[first_col].get("sum") is not None:
                total_val = desc[first_col]["sum"]
                kpis.append((f"Total {first_col.replace('_', ' ').title()}", f"{total_val:,.0f}"))
            elif first_col and desc[first_col].get("mean") is not None:
                avg = desc[first_col]["mean"]
                kpis.append((f"Avg {first_col.replace('_', ' ').title()}", f"{avg:,.1f}"))
    except Exception as e:
        logger.debug(f"KPI descriptive extraction skipped: {e}")

    # 3. Correlation KPI
    try:
        corr = results.get("correlation_analysis", {})
        if corr and "skipped_reason" not in corr:
            cols = list(corr.keys())
            if len(cols) >= 2:
                strongest = corr[cols[0]].get(cols[1])
                if strongest is not None:
                    kpis.append(("Correlation Strength", f"{strongest:.2f}"))
    except Exception as e:
        logger.debug(f"KPI correlation extraction skipped: {e}")

    # 4. Outlier KPI
    try:
        outliers = results.get("outlier_detection", {})
        if outliers and "skipped_reason" not in outliers:
            total_outliers = sum(v.get("outlier_count", 0) for v in outliers.values() if isinstance(v, dict))
            kpis.append(("Outliers Detected", str(total_outliers)))
    except Exception as e:
        logger.debug(f"KPI outlier extraction skipped: {e}")

    # Fallback if no specific metrics were found
    if not kpis:
        try:
            kpis.append(("Analyses Completed", str(len(results))))
        except Exception:
            pass

    return kpis[:4]


def _build_kpi_cards(kpis: List[Tuple[str, str]], styles: Any) -> List[Any]:
    """Renders extracted KPIs inside formatted summary cards."""
    if not kpis:
        return []
        
    elements = []
    row_size = 3
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

def _build_cover_page(profile: dict, styles: Any) -> List[Any]:
    """Constructs a formal title cover page."""
    elements = []
    elements.append(Spacer(1, 1.8 * inch))
    
    bar = Table([[""]], colWidths=[1.4 * inch], rowHeights=[0.08 * inch])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), THEME["gold"])]))
    elements.append(bar)
    elements.append(Spacer(1, 0.25 * inch))
    
    elements.append(Paragraph("AutoInsight AI", ParagraphStyle(
        name="CoverTitle", fontSize=32, leading=36, fontName="Helvetica-Bold", textColor=THEME["navy"])))
    
    elements.append(Paragraph("Automated Data Analysis Report", ParagraphStyle(
        name="CoverSubtitle", fontSize=14, leading=16, fontName="Helvetica", textColor=THEME["grey_mid"], spaceBefore=8)))
    
    elements.append(Spacer(1, 0.6 * inch))
    timestamp = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Prepared on {timestamp}", styles["Small"]))
    
    elements.append(Paragraph(
        f"Dataset: {profile.get('row_count', 'N/A')} rows &middot; {profile.get('column_count', 'N/A')} columns",
        styles["Small"]))
    
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
    elements.append(Paragraph("Executive Summary", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    plan_summary = plan.get("summary", "") if isinstance(plan, dict) else ""
    if plan_summary:
        elements.append(Paragraph(plan_summary, styles["Body"]))
        elements.append(Spacer(1, 0.12 * inch))

    if insights:
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

    snapshot_rows = [[
        Paragraph("Dimension", styles["TableHeader"]),
        Paragraph("Value", styles["TableHeader"]),
        Paragraph("Why it matters", styles["TableHeader"]),
    ]]

    snapshot_rows.append([
        Paragraph("Dataset size", styles["TableCell"]),
        Paragraph(f"{profile.get('row_count', 'N/A')} rows x {profile.get('column_count', 'N/A')} columns", styles["TableCell"]),
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

    if insights:
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

    overview_data = [
        [Paragraph("Metric", styles["TableHeader"]), Paragraph("Value", styles["TableHeader"])],
        [Paragraph("Total Rows", styles["TableCell"]), Paragraph(str(profile.get("row_count", "N/A")), styles["TableCell"])],
        [Paragraph("Total Columns", styles["TableCell"]), Paragraph(str(profile.get("column_count", "N/A")), styles["TableCell"])],
        [Paragraph("Numeric Columns", styles["TableCell"]), Paragraph(str(len(profile.get("numeric_columns", []))), styles["TableCell"])],
        [Paragraph("Categorical Columns", styles["TableCell"]), Paragraph(str(len(profile.get("categorical_columns", []))), styles["TableCell"])],
        [Paragraph("Datetime Columns", styles["TableCell"]), Paragraph(str(len(profile.get("datetime_columns", []))), styles["TableCell"])],
    ]

    table = Table(overview_data, colWidths=[3.3 * inch, 2.3 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))
    return elements


def _build_data_quality_section(validation_report: dict, cleaning_report: dict, styles: Any) -> List[Any]:
    """Formats data governance, health score, and cleaning operational metrics."""
    elements = []
    elements.append(Paragraph("Data Quality & Governance", styles["SectionHeading"]))
    elements.append(_section_divider())
    elements.append(Spacer(1, 0.1 * inch))

    health_score = validation_report.get("health_score", "N/A")
    elements.append(Paragraph(f"<b>Original Data Health Score:</b> {health_score} / 100", styles["Body"]))
    elements.append(Spacer(1, 0.08 * inch))

    quality_data = [
        [Paragraph("Check", styles["TableHeader"]), Paragraph("Result", styles["TableHeader"])],
        [Paragraph("Missing Cells (original)", styles["TableCell"]), Paragraph(str(validation_report.get("missing_cells", "N/A")), styles["TableCell"])],
        [Paragraph("Duplicate Rows (original)", styles["TableCell"]), Paragraph(str(validation_report.get("duplicate_rows", "N/A")), styles["TableCell"])],
        [Paragraph("Duplicates Removed", styles["TableCell"]), Paragraph(str(cleaning_report.get("duplicates_removed", "N/A")), styles["TableCell"])],
        [Paragraph("Rows After Cleaning", styles["TableCell"]), Paragraph(str(cleaning_report.get("rows_after_cleaning", "N/A")), styles["TableCell"])],
    ]

    table = Table(quality_data, colWidths=[3.3 * inch, 2.3 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))
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

    results = statistics.get("results", {}) if isinstance(statistics, dict) else {}
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


def _generate_recommendations(statistics: dict, validation_report: dict) -> List[str]:
    """Derives deterministic recommendations from statistical findings and health checks."""
    recs = []
    results = statistics.get("results", {}) if isinstance(statistics, dict) else {}
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
            
    health_score = validation_report.get("health_score", 100) if isinstance(validation_report, dict) else 100
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

    generated = visualizations.get("generated", []) if isinstance(visualizations, dict) else []
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


# ---------------------------------------------------------------------------
# LangGraph Node Implementation
# ---------------------------------------------------------------------------

def report_generator_node(state: GraphState) -> GraphState:
    """
    LangGraph pipeline node for deterministic PDF generation.
    
    Assembles profile, validation, statistical analysis, and visual image paths
    into a structured PDF report saved to disk.
    """
    logger.info("Report Generator Node started")

    profile = state.get("profile")
    if not profile:
        logger.error("Missing 'profile' in state for report generation.")
        raise ReportGeneratorError("Report Generator node requires 'profile' in state.")

    validation_report = state.get("validation_report") or state.get("validation") or state.get("validation_results")
    cleaning_report = state.get("cleaning_report") or state.get("cleaning") or state.get("cleaning_results")
    statistics = state.get("statistics")
    raw_insights = state.get("insights")
    if not validation_report or not cleaning_report or not statistics or raw_insights is None:
        raise ReportGeneratorError(
            "Report Generator node requires validation_report, cleaning_report, statistics, and insights in state."
        )

    plan = state.get("plan") or {}
    
    charts_state = state.get("visualizations") or state.get("charts") or []
    if isinstance(charts_state, dict):
        visualizations = charts_state
    elif isinstance(charts_state, list):
        visualizations = {"generated": charts_state}
    else:
        visualizations = {"generated": []}
        
    insights = raw_insights if isinstance(raw_insights, list) else [str(raw_insights)]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    file_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(REPORTS_DIR, file_name)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        report_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    kpis = _extract_kpi_highlights(statistics)

    story = []
    story += _build_cover_page(profile, styles)
    story += _build_title_section(styles)
    story += _build_kpi_cards(kpis, styles)
    story += _build_executive_snapshot(profile, plan, statistics, validation_report, styles)
    story += _build_key_findings_section(statistics, insights, validation_report, styles)
    story += _build_executive_summary(insights, plan, styles)
    story += _build_dataset_overview(profile, styles)
    story += _build_data_quality_section(validation_report, cleaning_report, styles)
    story.append(PageBreak())
    story += _build_statistics_section(statistics, styles)
    story += _build_recommendations_section(statistics, validation_report, styles)
    story += _build_charts_section(visualizations, styles)

    try:
        doc.build(story, onFirstPage=_add_page_footer, onLaterPages=_add_page_footer)
        logger.info(f"Report successfully compiled and saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to compile PDF report: {e}")
        raise ReportGeneratorError(f"Failed to build PDF report: {e}")

    return {**state, "report_path": report_path}