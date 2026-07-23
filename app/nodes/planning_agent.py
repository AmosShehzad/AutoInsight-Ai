import os
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.schemas.state import GraphState
from app.schemas.plan import AnalysisPlan
from utils.llm_factory import get_llm
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class PlanningAgentError(Exception):
    """Raised when the Planning Agent fails to produce a usable plan."""
    pass


def _enforce_profile_guardrails(plan_dict: dict, profile: dict) -> dict:
    """
    Apply deterministic safeguards so the plan remains aligned with profile facts.
    This protects against occasional generic LLM output.
    """
    numeric_cols = profile.get("numeric_columns", []) or []
    categorical_cols = profile.get("categorical_columns", []) or []
    categorical_stats = profile.get("categorical_stats", {}) or {}
    row_count = profile.get("row_count", 1) or 1

    # Edge Case Safeguard: If there are NO numeric columns at all, filter out numeric-only items
    if not numeric_cols:
        plan_dict["analyses"] = [
            a for a in plan_dict.get("analyses", [])
            if a not in ("descriptive_stats", "correlation_analysis", "trend_analysis", "outlier_detection", "grouped_summary_analysis")
        ]
        plan_dict["charts"] = [
            c for c in plan_dict.get("charts", [])
            if not isinstance(c, dict) or c.get("chart_type") not in ("line", "scatter", "histogram", "correlation_heatmap")
        ]
        if categorical_cols and "value_counts_analysis" not in plan_dict.get("analyses", []):
            plan_dict["analyses"].append("value_counts_analysis")

    filtered_charts = []
    for chart in plan_dict.get("charts", []):
        if not isinstance(chart, dict):
            filtered_charts.append(chart)
            continue

        cols = chart.get("columns", [])
        chart_type = chart.get("chart_type")

        # Block bar charts on numeric columns
        if chart_type == "bar" and cols and cols[0] in numeric_cols and cols[0] not in categorical_cols:
            continue

        # Block bar charts on categorical columns that are high-cardinality / unique identifiers
        # (more than 50% unique values relative to row count, e.g., Name, Ticket, ID)
        if chart_type == "bar" and cols and cols[0] in categorical_cols:
            unique_count = categorical_stats.get(cols[0], {}).get("unique_values", 0)
            if row_count > 0 and (unique_count / row_count) > 0.5:
                continue

        filtered_charts.append(chart)

    plan_dict["charts"] = filtered_charts

    analyses = list(plan_dict.get("analyses", []))
    charts = list(plan_dict.get("charts", []))

    datetime_cols = profile.get("datetime_columns", []) or []
    if datetime_cols and numeric_cols:
        analyses_text = " ".join(str(item).lower() for item in analyses)
        if "time" not in analyses_text and "trend" not in analyses_text:
            analyses.insert(0, f"trend_analysis_over_time_for_{datetime_cols[0]}")

        charts_text = " ".join(str(item).lower() for item in charts)
        if "line" not in charts_text and "time" not in charts_text:
            charts.insert(0, {
                "chart_type": "line",
                "columns": [datetime_cols[0]],
                "reason": f"Time series trend for {datetime_cols[0]}",
            })
            
    plan_dict["analyses"] = analyses
    plan_dict["charts"] = charts

    # If grouped_summary_analysis was chosen, request a bar chart of it too (if categorical columns exist)
    if "grouped_summary_analysis" in plan_dict.get("analyses", []) and numeric_cols:
        has_group_chart = any(isinstance(c, dict) and "group" in str(c.get("reason", "")).lower() for c in plan_dict["charts"])
        if not has_group_chart and categorical_cols:
            best_col = next((c for c in categorical_cols
                             if 2 <= profile.get("categorical_stats", {}).get(c, {}).get("unique_values", 0) <= 20), None)
            if best_col:
                plan_dict["charts"].append({
                    "chart_type": "bar",
                    "columns": [best_col],
                    "reason": f"Group comparison across {best_col}.",
                })

    # Ensure at least ONE histogram exists for a key numeric column if numeric columns exist
    has_histogram = any(isinstance(c, dict) and c.get("chart_type") == "histogram" for c in plan_dict["charts"])
    if not has_histogram and numeric_cols:
        plan_dict["charts"].append({
            "chart_type": "histogram",
            "columns": [numeric_cols[0]],
            "reason": f"Distribution of {numeric_cols[0]} across all records.",
        })

    # Ensure a correlation heatmap exists if 2+ numeric columns AND correlation_analysis was chosen
    if len(numeric_cols) >= 2 and "correlation_analysis" in plan_dict.get("analyses", []):
        has_heatmap = any(isinstance(c, dict) and c.get("chart_type") == "correlation_heatmap" for c in plan_dict["charts"])
        if not has_heatmap:
            plan_dict["charts"].append({
                "chart_type": "correlation_heatmap",
                "columns": numeric_cols,
                "reason": "Visual correlation strength between all numeric columns.",
            })

    # Keep chart count balanced
    num_analyses = len(plan_dict.get("analyses", []))
    max_charts = max(3, min(num_analyses + 2, 6))
    plan_dict["charts"] = plan_dict["charts"][:max_charts]

    return plan_dict


PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a data analysis planning expert. You will be given a JSON profile "
        "describing a dataset (column names, types, and basic statistics). "
        "Your job is to decide which statistical analyses and which charts make sense "
        "for THIS SPECIFIC dataset — not a generic list.\n\n"
        "Rules:\n"
        "- If there are datetime columns, prioritize trend/time-series analysis and line charts.\n"
        "- If there are multiple numeric columns, consider correlation analysis.\n"
        "- If a categorical column has very few unique values relative to row count, "
        "consider a bar chart of its value counts.\n"
        "- If a categorical column has very MANY unique values (like an ID or name column), "
        "do NOT suggest a bar chart for it — it would be useless.\n"
        "- Do not suggest more than 5 charts total.\n"
        "- Be specific: reference actual column names from the profile, never placeholders.\n\n"
        "IMPORTANT: only choose analyses from this exact list of supported names:\n"
        "'descriptive_stats', 'correlation_analysis', 'trend_analysis', 'outlier_detection', "
        "'value_counts_analysis', 'grouped_summary_analysis'. Do not invent new analysis names.\n\n"
        "If there is at least 1 numeric column AND 1 categorical column with a reasonable "
        "number of groups, ALWAYS include 'grouped_summary_analysis' — this answers real "
        "business questions like 'which category drives the most cost/revenue.'\n\n"
        "If you are given REVIEWER FEEDBACK from a previous attempt, you MUST address "
        "it directly — specifically include any analyses listed as missing, and do not "
        "repeat whatever the reviewer criticized.\n\n"
        "You MUST respond with a valid JSON object matching this exact shape:\n"
        '{{"analyses": ["..."], "charts": [{{"chart_type": "...", "columns": ["..."], "reason": "..."}}], "summary": "..."}}\n'
        "Return ONLY the JSON object, nothing else — no explanation, no markdown code fences."
    )),
    ("human", (
        "Dataset profile:\n{profile}\n\n"
        "{feedback_section}"
        "Produce the analysis plan now as a JSON object."
    )),
])


def _build_chain():
    """
    Helper that builds the prompt -> structured-LLM chain using the central factory.
    """
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(AnalysisPlan, method="json_mode")
    return PLANNING_PROMPT | structured_llm


def planning_agent_node(state: GraphState, max_retries: int = 2) -> dict:
    """
    LangGraph node (REAL AGENT — LLM-powered).
    Reads dataset profile and prior critic feedback (if any) from state,
    and returns a partial state dictionary updating state['plan'].
    """
    logger.info("Planning Agent Node started")
    profile = state.get("profile")
    critic_feedback = state.get("critic_feedback")

    if not profile:
        raise PlanningAgentError(
            "No 'profile' found in state. Planning Agent must run AFTER Profiling Node."
        )

    chain = _build_chain()
    profile_json = json.dumps(profile, indent=2)

    if critic_feedback and not critic_feedback.get("approved", True):
        feedback_section = (
            f"REVIEWER FEEDBACK from a previous attempt (you MUST address this):\n"
            f"Reason for rejection: {critic_feedback.get('reason', '')}\n"
            f"Specifically missing: {critic_feedback.get('missing_analyses', [])}\n\n"
        )
    else:
        feedback_section = ""

    last_error = None

    for attempt in range(1, max_retries + 2):
        try:
            plan_result: AnalysisPlan = chain.invoke({
                "profile": profile_json,
                "feedback_section": feedback_section,
            })
            plan_dict = plan_result.model_dump()

            plan_dict = _enforce_profile_guardrails(plan_dict, profile)
            plan_dict["_meta"] = {"attempts": attempt}

            logger.info(
                f"Planning Agent Node completed — produced plan with "
                f"{len(plan_dict.get('analyses', []))} analyses and {len(plan_dict.get('charts', []))} charts"
            )
            return {"plan": plan_dict}
        except Exception as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(1.5 * attempt)
                continue

    raise PlanningAgentError(
        f"Planning Agent failed after {max_retries + 1} attempts. Last error: {last_error}"
    )