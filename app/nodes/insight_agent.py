import os
import json
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.schemas.state import GraphState
from app.schemas.insight import InsightList

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
        "- Every insight MUST reference at least one actual number from the data "
        "(a percentage, an average, a count, a min/max — a real figure, not a vague claim).\n"
        "- NEVER write generic filler like 'the data shows interesting patterns' or "
        "'performance was strong' without a specific number attached.\n"
        "- Write in plain business English — a busy manager with no data background "
        "should understand every sentence instantly.\n"
        "- Produce between 3 and 7 insights total, not more.\n"
        "- Each insight must set 'source_analysis' to the exact analysis name it came from.\n"
        "- If a statistics result was skipped (has a 'skipped_reason'), do NOT invent an "
        "insight for it — simply don't write one for that analysis."
    )),
    ("human", (
        "Statistics results:\n{statistics}\n\n"
        "Charts generated:\n{charts}\n\n"
        "Write the business insights now."
    )),
]) 


def _build_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    structured_llm = llm.with_structured_output(InsightList)
    return INSIGHT_PROMPT | structured_llm


def insight_agent_node(state: GraphState, max_retries: int = 2) -> GraphState:
    """
    LangGraph node (REAL AGENT — LLM-powered).
    Runs AFTER Statistics and Visualization have both finished (fan-in),
    since it needs to read both of their results. Turns numbers and
    chart summaries into grounded, specific business language.

    Retry-with-backoff, same resilience pattern as the Planning Agent (Day 5).
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

    if not os.getenv("GROQ_API_KEY"):
        raise InsightAgentError(
            "GROQ_API_KEY not set. Add it to your .env file before running the Insight Agent."
        )

    # Build a short, LLM-friendly summary of the charts instead of dumping
    # full file paths (irrelevant noise for the LLM's reasoning).
    chart_summaries = [
        {"chart_type": c["chart_type"], "columns": c["columns"], "reason": c["reason"]}
        for c in visualizations.get("generated", [])
    ]

    chain = _build_chain()
    stats_json = json.dumps(statistics.get("results", {}), indent=2)
    charts_json = json.dumps(chart_summaries, indent=2)

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            result: InsightList = chain.invoke({
                "statistics": stats_json,
                "charts": charts_json,
            })
            insights_list = [insight.text for insight in result.insights]
            # Also keep the full structured version (with source_analysis) for
            # the Critic Agent to inspect later (Day 8).
            insights_detailed = [insight.model_dump() for insight in result.insights]

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