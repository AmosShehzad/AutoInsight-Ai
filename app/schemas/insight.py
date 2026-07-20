from pydantic import BaseModel, Field
from typing import List


class Insight(BaseModel):
    """
    ONE business insight, grounded in a specific piece of analysis.
    source_analysis tells us WHICH statistics result this insight came
    from — useful later for the Critic Agent (Day 8) to check insights
    are actually backed by real data, not made up.
    """
    text: str = Field(
        description=(
            "One clear, specific sentence in plain business language. "
            "MUST reference at least one actual number from the data "
            "(a percentage, count, average, etc.) — never generic filler."
        )
    )
    source_analysis: str = Field(
        description="Which analysis this insight is based on, e.g. 'trend_analysis', 'correlation_analysis'"
    )


class InsightList(BaseModel):
    """
    The full structured output of the Insight Generation Agent.
    A simple wrapper around a list — LangChain's structured output
    needs one top-level model, not a bare list.
    """
    insights: List[Insight] = Field(
        description="3 to 7 specific, numbers-grounded business insights about this dataset"
    )