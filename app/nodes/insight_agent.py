import os
import sys
from pathlib import Path

# Automatically adds the project root directory to Python's import search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.logger import get_logger
logger = get_logger(__name__)
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.schemas.state import GraphState
from app.schemas.insight import InsightList
from utils.llm_factory import get_llm

load_dotenv()


class InsightAgentError(Exception):
    """Raised when the Insight Agent fails to produce usable insights after retries."""
    pass


INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a business data analyst who writes executive-ready insights. "
        "You will be given the RESULTS of statistical analyses (real numbers) "
        "and a list of charts that were generated for this dataset.\n\n"
        "Rules:\n"
        "- Write EXACTLY 1 insight for each analysis that actually produced results "
        "(skip analyses with a 'skipped_reason' or 'error').\n"
        "- If an analysis covers multiple columns (like descriptive_stats with 2 numeric "
        "columns), you may write up to 2 insights for it if genuinely distinct, but never more.\n"
        "- Total insights must be between 3 and 6 — if you'd naturally write more, pick "
        "the most important ones.\n"
        "- Every insight MUST reference at least one actual number from the data.\n"
        "- NEVER write generic filler.\n"
        "- Write in plain business English.\n"
        "- Each insight must set 'source_analysis' to the exact analysis name it came from.\n"
        "- Write plain numbers only — do NOT use currency symbols like $ or special characters.\n\n"
        "You MUST respond with a valid JSON object matching this exact shape:\n"
        '{{"insights": [{{"text": "...", "source_analysis": "..."}}, ...]}}\n'
        "Return ONLY the JSON object, nothing else — no explanation, no markdown code fences."
    )),
    ("human", (
        "Statistics results:\n{statistics}\n\n"
        "Charts generated:\n{charts}\n\n"
        "Write the business insights now, as a JSON object."
    )),
])


def _build_chain(temperature: float = 0.3):
    """
    Helper that builds the prompt -> structured-LLM chain using the central factory.
    """
    llm = get_llm(temperature=temperature)
    # Switched to JSON mode for improved structured output reliability
    structured_llm = llm.with_structured_output(InsightList, method="json_mode")
    return INSIGHT_PROMPT | structured_llm


def _dedupe_similar_insights(insights: list) -> list:
    """
    Removes insights that are near-duplicates of an earlier one — simple
    check based on shared source_analysis + overlapping key words.
    Keeps the FIRST occurrence, drops later similar ones.
    """
    seen_sources = {}
    deduped = []
    for insight in insights:
        source = insight.source_analysis
        # allow max 2 insights per source_analysis — if already at 2, skip
        count = seen_sources.get(source, 0)
        if count >= 2:
            continue
        seen_sources[source] = count + 1
        deduped.append(insight)
    return deduped


def insight_agent_node(state: GraphState, max_retries: int = 2) -> GraphState:
    """
    LangGraph node (REAL AGENT — LLM-powered).
    Turns numerical results and chart summaries into grounded business language.
    """
    statistics = state.get("statistics")
    visualizations = state.get("visualizations")

    if not statistics:
        raise InsightAgentError(
            "No 'statistics' found in state. Insight Agent must run AFTER the Statistics Node."
        )
    if not visualizations:
        raise InsightAgentError(
            "No 'visualizations' found in state. Insight Agent must run AFTER the Visualization Node."
        )

    chart_summaries = [
        {"chart_type": c["chart_type"], "columns": c["columns"], "reason": c["reason"]}
        for c in visualizations.get("generated", [])
    ]

    stats_json = json.dumps(statistics.get("results", {}), indent=2)
    charts_json = json.dumps(chart_summaries, indent=2)

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            # Slightly vary temperature on each retry attempt — helps the
            # model escape a bad generation pattern instead of repeating
            # the exact same malformed output every time.
            chain = _build_chain(temperature=0.3 + (attempt - 1) * 0.15)
            result: InsightList = chain.invoke({
                "statistics": stats_json,
                "charts": charts_json,
            })
            deduped_insights = _dedupe_similar_insights(result.insights)
            insights_list = [insight.text for insight in deduped_insights]
            insights_detailed = [insight.model_dump() for insight in deduped_insights]

            return {
                "insights": insights_list,
                "insights_detailed": insights_detailed,
            }
        except Exception as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(1.5 * attempt)
                continue

    raise InsightAgentError(
        f"Insight Agent failed after {max_retries + 1} attempts. Last error: {last_error}"
    )