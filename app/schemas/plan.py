from pydantic import BaseModel, Field
from typing import List


class ChartRequest(BaseModel):
    """
    Describes ONE chart the Planning Agent wants generated.
    Using a nested model (not just a string) lets the agent be specific
    about WHICH columns go into WHICH chart type.
    """
    chart_type: str = Field(
        description="Type of chart, e.g. 'histogram', 'bar', 'line', 'scatter', 'correlation_heatmap'"
    )
    columns: List[str] = Field(
        description="Which column(s) from the dataset this chart should use"
    )
    reason: str = Field(
        description="One short sentence explaining why this chart is useful for THIS dataset"
    )


class AnalysisPlan(BaseModel):
    """
    The full structured output of the Planning Agent.
    This exact shape is what gets stored in GraphState['plan'],
    and what the Statistics Node (Day 5) and Visualization Node (Day 6)
    will read and act on.
    """
    analyses: List[str] = Field(
        description=(
            "List of statistical methods to run, e.g. 'descriptive_stats', "
            "'correlation_analysis', 'trend_analysis', 'outlier_detection'"
        )
    )
    charts: List[ChartRequest] = Field(
        description="List of charts to generate, each with type, columns, and reason"
    )
    summary: str = Field(
        description="1-2 sentence plain-English summary of the overall analysis strategy for this dataset"
    )