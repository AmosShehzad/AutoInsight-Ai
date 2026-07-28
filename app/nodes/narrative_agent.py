import json
import time
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from app.config import config
from app.schemas.state import GraphState
from app.schemas.narrative import ExecutiveNarrative
from app.node_wrapper import node_error_boundary
from app.logger import get_logger

logger = get_logger(__name__)


class NarrativeAgentError(Exception):
    pass


NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior management consultant writing for a business owner who is NOT technical. "
        "You will be given statistics, correlations, and insights already computed from a real dataset.\n\n"
        "Rules:\n"
        "- NEVER invent a number that isn't in the data you were given.\n"
        "- Write like a consultant telling a story, not a scientist reporting results.\n"
        "- Every claim must be traceable to a real number you were given.\n"
        "- Do not mention forecasts, predictions, or future performance — you only have historical data.\n"
        "- Column intelligence: only cover the TOP 5 most important numeric columns, not every column.\n"
        "- Confidence ratings must reflect real signal strength: a correlation above 0.7 is High confidence, "
        "0.4-0.7 is Medium, below 0.4 is Low.\n\n"
        "You MUST respond with a valid JSON object using EXACTLY this structure and these EXACT field "
        "names — do not rename, add, or remove any field:\n\n"
        "{{\n"
        '  "business_story": "a 3-5 sentence paragraph",\n'
        '  "column_intelligence": [\n'
        "    {{\n"
        '      "column": "exact column name",\n'
        '      "purpose": "one sentence: what this column represents",\n'
        '      "business_interpretation": "one sentence: what the pattern means",\n'
        '      "risk_note": "one sentence: a caution, or the exact text \'None identified.\'"\n'
        "    }}\n"
        "  ],\n"
        '  "opportunities": [\n'
        "    {{\n"
        '      "title": "short opportunity name",\n'
        '      "recommendation": "what action to take",\n'
        '      "expected_impact": "plain-language expected result",\n'
        '      "confidence": "High, Medium, or Low"\n'
        "    }}\n"
        "  ],\n"
        '  "risks": [\n'
        "    {{\n"
        '      "title": "short risk name",\n'
        '      "description": "what the risk is",\n'
        '      "priority": "Critical, High, Medium, or Low"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "opportunities and risks MUST be arrays of OBJECTS with those exact fields — "
        "NEVER plain strings. column_intelligence items MUST have purpose, business_interpretation, "
        "and risk_note — NEVER importance, confidence, or correlation as field names.\n\n"
        "Return ONLY the JSON object, no explanation, no markdown fences."
    )),
    ("human", (
        "Profile:\n{profile}\n\n"
        "Statistics:\n{statistics}\n\n"
        "Existing insights:\n{insights}\n\n"
        "Write the executive narrative now, as JSON matching the EXACT structure shown above."
    )),
])


def _repair_narrative_json(raw: dict) -> dict:
    """
    Defensive repair layer: if the LLM's JSON is close but not exact
    (e.g., opportunities as plain strings instead of objects), fix the
    common cases here instead of failing validation and burning a retry.
    """
    repaired = dict(raw)

    # Fix opportunities: if items are strings, wrap them into the expected shape
    fixed_opps = []
    for item in repaired.get("opportunities", []):
        if isinstance(item, str):
            fixed_opps.append({
                "title": item[:60],
                "recommendation": item,
                "expected_impact": "Not specified.",
                "confidence": "Medium",
            })
        elif isinstance(item, dict):
            fixed_opps.append(item)
    repaired["opportunities"] = fixed_opps

    # Fix risks: same pattern
    fixed_risks = []
    for item in repaired.get("risks", []):
        if isinstance(item, str):
            fixed_risks.append({
                "title": item[:60],
                "description": item,
                "priority": "Medium",
            })
        elif isinstance(item, dict):
            fixed_risks.append(item)
    repaired["risks"] = fixed_risks

    # Fix column_intelligence: map common wrong field names to the expected ones
    fixed_cols = []
    for item in repaired.get("column_intelligence", []):
        if not isinstance(item, dict):
            continue
        fixed_cols.append({
            "column": item.get("column", "Unknown"),
            "purpose": item.get("purpose") or item.get("importance") or "Not specified.",
            "business_interpretation": item.get("business_interpretation") or item.get("confidence") or "Not specified.",
            "risk_note": item.get("risk_note") or (str(item.get("correlation")) if item.get("correlation") else "None identified."),
        })
    repaired["column_intelligence"] = fixed_cols

    return repaired


def _build_raw_chain(temperature: float = 0.3):
    """Returns the LLM chain without structured output to allow manual JSON repair."""
    llm = get_llm(temperature=temperature, tags=["narrative_agent"])
    return NARRATIVE_PROMPT | llm


@node_error_boundary("narrative_agent")
def narrative_agent_node(state: GraphState) -> dict:
    profile = state.get("profile")
    statistics = state.get("statistics")
    insights = state.get("insights")

    if not profile or not statistics:
        raise NarrativeAgentError("Narrative Agent needs 'profile' and 'statistics' in state.")

    chain = _build_raw_chain()
    inputs = {
        "profile": json.dumps({k: v for k, v in profile.items() if k != "categorical_stats"}, indent=2)[:3000],
        "statistics": json.dumps(statistics.get("results", {}), indent=2)[:4000],
        "insights": json.dumps(insights or [], indent=2),
    }

    last_error = None
    for attempt in range(1, config.MAX_LLM_RETRIES + 2):
        try:
            raw_response = chain.invoke(inputs)
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

            # Clean markdown code fences if present
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            # Parse, repair structure, and validate with Pydantic
            raw_dict = json.loads(raw_text)
            repaired_dict = _repair_narrative_json(raw_dict)
            result = ExecutiveNarrative(**repaired_dict)

            logger.info(f"Narrative Agent completed on attempt {attempt}")
            return {"narrative": result.model_dump()}

        except Exception as e:
            last_error = e
            logger.warning(f"Narrative Agent attempt {attempt} failed validation: {e}")
            if attempt <= config.MAX_LLM_RETRIES:
                time.sleep(1.5 * attempt)
                continue

    raise NarrativeAgentError(f"Narrative Agent failed after retries: {last_error}")