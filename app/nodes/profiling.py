import pandas as pd
from app.schemas.state import GraphState


def profile_data_node(state: GraphState) -> GraphState:
    """
    LangGraph node: inspects the cleaned dataframe and builds a profile —
    column types, descriptive stats, and a numeric/categorical split.
    This profile is what the Planning Agent (Day 4) will read to decide
    what analysis makes sense.
    """
    df: pd.DataFrame = state["cleaned_dataframe"]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()) if len(df) > 1 else 0.0,
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "unique_values": int(df[col].nunique()),
        }

    categorical_stats = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(5).to_dict()
        categorical_stats[col] = {
            "unique_values": int(df[col].nunique()),
            "top_5_values": {str(k): int(v) for k, v in value_counts.items()},
        }

    profile = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }

    return {**state, "profile": profile}