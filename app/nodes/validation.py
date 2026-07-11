import pandas as pd
from app.schemas.state import GraphState


def validate_data_node(state: GraphState) -> GraphState:
    """
    LangGraph node: inspects the DataFrame in state for missing values,
    duplicates, and dtype issues, and calculates a simple health score.
    """
    df: pd.DataFrame = state["dataframe"]

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    missing_percent = round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())

    dtype_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Simple health score: starts at 100, penalized by missing data and duplicates
    health_score = 100.0
    health_score -= missing_percent          # missing data hurts score directly
    duplicate_percent = round((duplicate_rows / len(df)) * 100, 2) if len(df) else 0.0
    health_score -= duplicate_percent
    health_score = max(0.0, round(health_score, 2))  # never go below 0

    validation_report = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "dtype_summary": dtype_summary,
        "health_score": health_score,
    }

    return {**state, "validation_report": validation_report}