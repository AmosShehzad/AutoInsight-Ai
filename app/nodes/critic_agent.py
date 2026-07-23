import os
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from app.logger import get_logger
logger = get_logger(__name__)
from app.schemas.state import GraphState
from app.schemas.critic import CriticFeedback
from utils.llm_factory import get_llm

load_dotenv()

MAX_REVISIONS = 2   # Safety valve — after this many loops, force approval


class CriticAgentError(Exception):
    """Raised when the Critic Agent fails to produce a usable judgment after retries."""
    pass


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict but fair data analysis reviewer. You will be given the full "
        "output of an automated analysis pipeline: the analysis PLAN, the STATISTICS "
        "that were computed, the CHARTS that were generated, and the business INSIGHTS "
        "that were written.\n\n"
        "Your job: decide if this analysis is thorough and ready to go into an "
        "executive report, or if it needs more work.\n\n"
        "Approve (approved=true) if:\n"
        "- The statistics that ran are relevant to the dataset's structure\n"
        "- The insights are specific and reference real numbers\n"
        "- There isn't an obvious, important gap in the analysis\n\n"
        "Reject (approved=false) if:\n"
        "- Key analyses were skipped that clearly should have run given the data "
        "(check the 'unsupported_analyses' and 'skipped_reason' fields)\n"
        "- Insights are vague or don't match the actual statistics\n"
        "- Something important and obviously relevant to this data is missing entirely\n\n"
        "Be specific in your reason. If rejecting, list the exact analysis names that "
        "should be added in missing_analyses — only use names from this list: "
        "'descriptive_stats', 'correlation_analysis', 'trend_analysis', 'outlier_detection', "
        "'value_counts_analysis'.\n\n"
        "You MUST respond with a valid JSON object matching this exact shape:\n"
        '{{"approved": true, "reason": "...", "missing_analyses": ["..."]}}\n'
        "Return ONLY the JSON object, nothing else — no explanation, no markdown code fences."
    )),
    ("human", (
        "Plan:\n{plan}\n\n"
        "Statistics results:\n{statistics}\n\n"
        "Charts generated:\n{charts}\n\n"
        "Insights:\n{insights}\n\n"
        "Review this analysis now, and output your decision as a JSON object."
    )),
])


def _build_chain():
    """
    Helper that builds the prompt -> structured-LLM chain using the central factory.
    """
    llm = get_llm(temperature=0.1)
    # Switched to JSON mode for improved structured output reliability
    structured_llm = llm.with_structured_output(CriticFeedback, method="json_mode")
    return CRITIC_PROMPT | structured_llm


def critic_agent_node(state: GraphState, max_retries: int = 2) -> GraphState:
    """
    LangGraph node (REAL AGENT — LLM-powered).
    Reviews the full analysis produced so far and decides if it's acceptable.
    """
    plan = state.get("plan")
    statistics = state.get("statistics")
    visualizations = state.get("visualizations")
    insights = state.get("insights")
    revision_count = state.get("revision_count", 0)

    if not plan or not statistics or not visualizations or insights is None:
        raise CriticAgentError(
            "Critic Agent needs 'plan', 'statistics', 'visualizations', and 'insights' "
            "all present in state. It must run AFTER the Insight Agent."
        )

    if revision_count >= MAX_REVISIONS:
        forced_feedback = {
            "approved": True,
            "reason": f"Forced approval after reaching max revision limit ({MAX_REVISIONS}).",
            "missing_analyses": [],
        }
        return {"critic_feedback": forced_feedback}

    chart_summaries = [
        {"chart_type": c["chart_type"], "columns": c["columns"]}
        for c in visualizations.get("generated", [])
    ]

    chain = _build_chain()
    inputs = {
        "plan": json.dumps(plan, indent=2),
        "statistics": json.dumps(statistics, indent=2),
        "charts": json.dumps(chart_summaries, indent=2),
        "insights": json.dumps(insights, indent=2),
    }

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            result: CriticFeedback = chain.invoke(inputs)
            feedback_dict = result.model_dump()
            return {"critic_feedback": feedback_dict}
        except Exception as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(1.5 * attempt)
                continue

    raise CriticAgentError(
        f"Critic Agent failed after {max_retries + 1} attempts. Last error: {last_error}"
    )