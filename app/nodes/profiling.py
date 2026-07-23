"""
Profiling Node for AutoInsight AI
Inspects the cleaned dataframe and builds a deep statistical profile —
column dtypes, distribution metrics, categorical frequencies, and time spans.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from app.logger import get_logger
from app.schemas.state import GraphState

logger = get_logger(__name__)


class DataProfilingError(Exception):
    """Raised when dataset profiling fails."""
    pass


def _sanitize_val(val: Any) -> Optional[Any]:
    """Ensures values are JSON-serializable standard Python types (no NaN or Inf)."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    return str(val)


def profile_data_node(state: GraphState) -> GraphState:
    """
    LangGraph Node: Profiling node that reads the cleaned dataframe
    (falling back to raw dataframe if cleaning was skipped) and constructs
    a comprehensive statistical profile for downstream AI planners.
    """
    df: pd.DataFrame = state.get("cleaned_dataframe")
    if df is None:
        logger.info("Property 'cleaned_dataframe' not found in state. Falling back to 'dataframe'.")
        df = state.get("dataframe")

    if df is None:
        logger.error("Profiling Node failed: Neither 'cleaned_dataframe' nor 'dataframe' found in state.")
        raise DataProfilingError("No valid DataFrame found in state for profiling.")

    total_rows, total_cols = df.shape
    logger.info(f"Profiling Node started — {total_rows} rows, {total_cols} columns")

    # Column classification
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns]
    categorical_cols = [str(col) for col in df.select_dtypes(include=["object", "string", "category"]).columns]
    datetime_cols = [
        str(col) for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    # -----------------------------------------------------------------------
    # Numeric Statistics
    # -----------------------------------------------------------------------
    numeric_stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            numeric_stats[col] = {"skipped_reason": "All values are NaN"}
            continue

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        
        numeric_stats[col] = {
            "mean": _sanitize_val(series.mean()),
            "median": _sanitize_val(series.median()),
            "std": _sanitize_val(series.std() if len(series) > 1 else 0.0),
            "min": _sanitize_val(series.min()),
            "q25": _sanitize_val(q25),
            "q75": _sanitize_val(q75),
            "iqr": _sanitize_val(q75 - q25),
            "max": _sanitize_val(series.max()),
            "skewness": _sanitize_val(series.skew() if len(series) > 2 else 0.0),
            "unique_values": int(series.nunique()),
            "zero_count": int((series == 0).sum()),
            "zero_percent": round(float((series == 0).mean() * 100.0), 2),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percent": round(float(df[col].isnull().mean() * 100.0), 2),
        }

    # -----------------------------------------------------------------------
    # Categorical Statistics
    # -----------------------------------------------------------------------
    categorical_stats = {}
    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            categorical_stats[col] = {"skipped_reason": "All values are NaN"}
            continue

        unique_count = int(series.nunique())
        top_5 = series.value_counts().head(5).to_dict()
        cardinality_ratio = round(float(unique_count / len(series)), 4) if len(series) else 0.0

        categorical_stats[col] = {
            "unique_values": unique_count,
            "cardinality_ratio": cardinality_ratio,
            "is_high_cardinality": bool(cardinality_ratio > 0.4 and unique_count > 20),
            "most_frequent": _sanitize_val(series.mode().iloc[0]) if not series.mode().empty else None,
            "top_5_values": {str(k): int(v) for k, v in top_5.items()},
            "missing_count": int(df[col].isnull().sum()),
            "missing_percent": round(float(df[col].isnull().mean() * 100.0), 2),
        }

    # -----------------------------------------------------------------------
    # Datetime Statistics
    # -----------------------------------------------------------------------
    datetime_stats = {}
    for col in datetime_cols:
        series = df[col].dropna()
        if series.empty:
            datetime_stats[col] = {"skipped_reason": "All values are NaN"}
            continue

        min_date = series.min()
        max_date = series.max()
        time_span_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else None

        datetime_stats[col] = {
            "min_date": _sanitize_val(min_date),
            "max_date": _sanitize_val(max_date),
            "timespan_days": time_span_days,
            "unique_dates": int(series.nunique()),
            "missing_count": int(df[col].isnull().sum()),
        }

    # -----------------------------------------------------------------------
    # Assembled Profile Payload
    # -----------------------------------------------------------------------
    profile = {
        "row_count": total_rows,
        "column_count": total_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "datetime_stats": datetime_stats,
    }

    logger.info(
        f"Profiling complete — Analyzed {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical, and {len(datetime_cols)} datetime column(s)."
    )

    return {**state, "profile": profile}