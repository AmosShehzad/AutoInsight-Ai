from typing import TypedDict, Optional
import pandas as pd


class GraphState(TypedDict, total=False):
    """
    Shared 'memory' that flows through every node in our LangGraph graph.
    Grows with each day as we add more nodes.
    """
    file_path: str
    dataframe: pd.DataFrame
    validation_report: dict
    cleaned_dataframe: pd.DataFrame
    cleaning_report: dict
    profile: dict