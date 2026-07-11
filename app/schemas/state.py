from typing import TypedDict, Optional
import pandas as pd


class GraphState(TypedDict, total=False):
    """
    This is the shared 'memory' that flows through every node in our LangGraph graph.
    Every node reads from this and writes back into it.
    We will keep ADDING fields to this as we build more nodes (Day 2, 3, 4...).
    """
    file_path: str
    dataframe: pd.DataFrame
    validation_report: dict