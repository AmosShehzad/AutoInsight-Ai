from pydantic import BaseModel, Field
from typing import List


class CriticFeedback(BaseModel):
    """
    The Critic Agent's judgment on the full analysis produced so far.
    This exact shape drives the conditional edge in graph.py — 'approved'
    decides whether the graph moves forward or loops back to Planning.
    """
    approved: bool = Field(
        description="True if the analysis is thorough and ready for the report. False if it needs more work."
    )
    reason: str = Field(
        description="1-3 sentences explaining the decision. If not approved, be SPECIFIC about what's missing or weak."
    )
    missing_analyses: List[str] = Field(
        default_factory=list,
        description=(
            "If not approved, list specific analysis types that should be added "
            "(e.g. 'outlier_detection', 'correlation_analysis'). Empty list if approved."
        )
    )