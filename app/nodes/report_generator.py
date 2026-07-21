import os
import uuid
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

from app.schemas.state import GraphState

REPORTS_DIR = "outputs/reports"


class ReportGeneratorError(Exception):
    """Raised when the PDF report cannot be generated."""
    pass


# ---------------------------------------------------------------------------
# Each function below builds ONE section of the report and returns a list
# of "flowables" (ReportLab's term for a piece of content to lay out).
# Small, single-purpose functions — same pattern as statistics.py and
# visualization.py — makes Day 10's polish work section-by-section, easy.
# ---------------------------------------------------------------------------

def _build_title_section(styles) -> list:
    """Report title + generation timestamp."""
    elements = []
    elements.append(Paragraph("AutoInsight AI — Data Analysis Report", styles["ReportTitle"]))
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    elements.append(Paragraph(f"Generated on {timestamp}", styles["Small"]))
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_executive_summary(insights: list, plan: dict, styles) -> list:
    """
    Executive summary: the plan's own summary sentence, followed by the
    top insights — this is the section a busy manager reads first (and
    maybe only).
    """
    elements = []
    elements.append(Paragraph("Executive Summary", styles["SectionHeading"]))

    plan_summary = plan.get("summary", "")
    if plan_summary:
        elements.append(Paragraph(plan_summary, styles["Body"]))
        elements.append(Spacer(1, 0.15 * inch))

    if insights:
        for insight in insights:
            elements.append(Paragraph(f"• {insight}", styles["Body"]))
    else:
        elements.append(Paragraph("No insights were generated for this dataset.", styles["Body"]))

    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_dataset_overview(profile: dict, styles) -> list:
    """Basic shape of the dataset: row/column counts and column type breakdown."""
    elements = []
    elements.append(Paragraph("Dataset Overview", styles["SectionHeading"]))

    overview_data = [
        ["Metric", "Value"],
        ["Total Rows", str(profile.get("row_count", "N/A"))],
        ["Total Columns", str(profile.get("column_count", "N/A"))],
        ["Numeric Columns", str(len(profile.get("numeric_columns", [])))],
        ["Categorical Columns", str(len(profile.get("categorical_columns", [])))],
        ["Datetime Columns", str(len(profile.get("datetime_columns", [])))],
    ]

    table = Table(overview_data, colWidths=[2.5 * inch, 2.5 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_data_quality_section(validation_report: dict, cleaning_report: dict, styles) -> list:
    """
    Data quality: what the raw data looked like BEFORE cleaning (validation_report)
    and what was actually done to fix it (cleaning_report) — full transparency,
    nothing hidden from the reader about how the data was modified.
    """
    elements = []
    elements.append(Paragraph("Data Quality", styles["SectionHeading"]))

    health_score = validation_report.get("health_score", "N/A")
    elements.append(Paragraph(f"Original Data Health Score: {health_score}/100", styles["Body"]))

    quality_data = [
        ["Check", "Result"],
        ["Missing Cells (original)", str(validation_report.get("missing_cells", "N/A"))],
        ["Duplicate Rows (original)", str(validation_report.get("duplicate_rows", "N/A"))],
        ["Duplicates Removed", str(cleaning_report.get("duplicates_removed", "N/A"))],
        ["Rows After Cleaning", str(cleaning_report.get("rows_after_cleaning", "N/A"))],
    ]

    table = Table(quality_data, colWidths=[3 * inch, 2 * inch])
    table.setStyle(_default_table_style())
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    return elements


def _build_statistics_section(statistics: dict, styles) -> list:
    """
    Statistical summary: dumps each completed analysis's results as
    readable text. Rough on purpose today — Day 10 will format this
    into cleaner tables per analysis type.
    """
    elements = []
    elements.append(Paragraph("Statistical Summary", styles["SectionHeading"]))

    results = statistics.get("results", {})
    if not results:
        elements.append(Paragraph("No statistical analyses were completed.", styles["Body"]))

    for analysis_name, analysis_result in results.items():
        elements.append(Paragraph(analysis_name.replace("_", " ").title(), styles["SubHeading"]))

        if "skipped_reason" in analysis_result:
            elements.append(Paragraph(f"Skipped: {analysis_result['skipped_reason']}", styles["Body"]))
        elif "error" in analysis_result:
            elements.append(Paragraph(f"Error: {analysis_result['error']}", styles["Body"]))
        else:
            # Rough dump of the dict as readable text — good enough for today's "ugly is fine" goal
            summary_text = str(analysis_result)
            elements.append(Paragraph(summary_text, styles["Small"]))

        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Spacer(1, 0.2 * inch))
    return elements


def _build_charts_section(visualizations: dict, styles) -> list:
    """
    Pulls in every successfully generated chart image and embeds it into
    the PDF, with its reason as a caption underneath.
    """
    elements = []
    elements.append(Paragraph("Charts", styles["SectionHeading"]))

    generated = visualizations.get("generated", [])
    if not generated:
        elements.append(Paragraph("No charts were generated for this dataset.", styles["Body"]))
        return elements

    for chart in generated:
        file_path = chart.get("file_path")
        if not file_path or not os.path.exists(file_path):
            # Defensive check — don't crash the whole report if one chart file is missing
            elements.append(Paragraph(f"[Chart image missing: {chart.get('chart_type')}]", styles["Body"]))
            continue

        # Fixed width, auto height — keeps every chart a consistent size on the page
        img = Image(file_path, width=5.5 * inch, height=3.4 * inch)
        elements.append(img)
        elements.append(Paragraph(chart.get("reason", ""), styles["Caption"]))
        elements.append(Spacer(1, 0.25 * inch))

    return elements


def _default_table_style() -> TableStyle:
    """Shared styling for every table in the report — one place to tweak the look later."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])


def _build_styles():
    """
    Defines the custom paragraph styles used throughout the report.
    Built once per report generation, based on ReportLab's default stylesheet.
    """
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=24, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=14, leading=18, fontName="Helvetica-Bold",
                               spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#2C3E50")))
    styles.add(ParagraphStyle(name="SubHeading", fontSize=11, leading=14, fontName="Helvetica-Bold",
                               spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=11, textColor=colors.grey))
    styles.add(ParagraphStyle(name="Caption", fontSize=8, leading=11, textColor=colors.grey,
                               alignment=1))  # centered
    return styles


def report_generator_node(state: GraphState) -> GraphState:
    """
    LangGraph node (PIPELINE NODE — deterministic, no LLM call).
    Assembles everything the graph has produced into a single PDF report.
    Runs only after the Critic Agent has approved the analysis.
    """
    profile = state.get("profile")
    validation_report = state.get("validation_report")
    cleaning_report = state.get("cleaning_report")
    plan = state.get("plan")
    statistics = state.get("statistics")
    visualizations = state.get("visualizations")
    insights = state.get("insights")

    required = {
        "profile": profile, "validation_report": validation_report,
        "cleaning_report": cleaning_report, "plan": plan,
        "statistics": statistics, "visualizations": visualizations,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ReportGeneratorError(
            f"Report Generator is missing required state fields: {missing}. "
            "It must run after the full pipeline has completed."
        )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    file_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(REPORTS_DIR, file_name)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        report_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    # Build the full "story" — a flat list of flowables in the order
    # they should appear on the page. ReportLab handles page breaks
    # automatically as content overflows.
    story = []
    story += _build_title_section(styles)
    story += _build_executive_summary(insights or [], plan, styles)
    story += _build_dataset_overview(profile, styles)
    story += _build_data_quality_section(validation_report, cleaning_report, styles)
    story.append(PageBreak())
    story += _build_statistics_section(statistics, styles)
    story += _build_charts_section(visualizations, styles)

    try:
        doc.build(story)
    except Exception as e:
        raise ReportGeneratorError(f"Failed to build PDF report: {e}")

    return {"report_path": report_path}