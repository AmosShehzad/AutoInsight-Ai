import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.schemas.state import GraphState
from app.schemas.plan import AnalysisPlan

# Load environment variables (GROQ_API_KEY) from .env into os.environ
load_dotenv()


class PlanningAgentError(Exception):
    """Raised when the Planning Agent fails to produce a usable plan."""
    pass


def _enforce_profile_guardrails(plan_dict: dict, profile: dict) -> dict:
    """
    Apply deterministic safeguards so the plan remains aligned with profile facts.
    This protects against occasional generic LLM output.
    """
    analyses = list(plan_dict.get("analyses", []))
    charts = list(plan_dict.get("charts", []))

    datetime_cols = profile.get("datetime_columns", []) or []
    if datetime_cols:
        analyses_text = " ".join(str(item).lower() for item in analyses)
        if "time" not in analyses_text and "trend" not in analyses_text:
            analyses.insert(0, f"trend_analysis_over_time_for_{datetime_cols[0]}")

        charts_text = " ".join(str(item).lower() for item in charts)
        if "line" not in charts_text and "time" not in charts_text:
            charts.insert(0, f"line_chart_{datetime_cols[0]}")

    # Keep output within the prompt's max chart count rule.
    plan_dict["analyses"] = analyses
    plan_dict["charts"] = charts[:5]
    return plan_dict


# --- Prompt template ---
# This is the actual "instructions" the LLM sees every time this node runs.
# {profile} gets filled in at runtime with the real dataset profile (as JSON text).
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
        "- Be specific: reference actual column names from the profile, never placeholders."
    )),
    ("human", "Dataset profile:\n{profile}\n\nProduce the analysis plan now."),
])


def planning_agent_node(state: GraphState) -> GraphState:
    """
    LangGraph node (REAL AGENT — LLM-powered).
    Reads the dataset profile from state, asks the LLM to reason about it,
    and writes a structured AnalysisPlan back into state as a plain dict.
    """
    profile = state.get("profile")
    if not profile:
        raise PlanningAgentError(
            "No 'profile' found in state. Planning Agent must run AFTER Profiling Node."
        )

    if not os.getenv("GROQ_API_KEY"):
        raise PlanningAgentError(
            "GROQ_API_KEY not set. Add it to your .env file before running the Planning Agent."
        )

    # Set up the LLM. temperature=0.2 keeps output focused and consistent
    # rather than creative/random — we want reliable planning, not variety.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    # .with_structured_output() is the key LangChain feature here:
    # it forces the LLM's response to match our AnalysisPlan Pydantic schema,
    # and automatically parses the JSON reply into a validated AnalysisPlan object.
    structured_llm = llm.with_structured_output(AnalysisPlan)

    # Chain the prompt template into the structured LLM using LCEL (LangChain
    # Expression Language) — the "|" pipe operator. This means:
    # "take the prompt, fill in the variables, send it to the LLM, parse the result."
    chain = PLANNING_PROMPT | structured_llm

    try:
        plan_result: AnalysisPlan = chain.invoke({
            "profile": json.dumps(profile, indent=2)
        })
    except Exception as e:
        raise PlanningAgentError(f"Planning Agent failed to produce a valid plan: {e}")

    # Convert the validated Pydantic object into a plain dict before
    # storing in state — safer for LangGraph's checkpoint serialization.
    plan_dict = plan_result.model_dump()
    plan_dict = _enforce_profile_guardrails(plan_dict, profile)

    return {**state, "plan": plan_dict}