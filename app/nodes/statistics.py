"""
Statistics Node Module for AutoInsight AI
Executes statistical analyses based on the execution plan.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from app.logger import get_logger
from app.node_wrapper import node_error_boundary

logger = get_logger(__name__)


class StatisticsNodeError(Exception):
    """Raised when the Statistics Node cannot execute with the provided state."""
    pass


def _get_dataframe(state: dict) -> pd.DataFrame:
    """
    Reads the cleaned dataframe from state using the CANONICAL key only.
    No silent fallbacks — if 'cleaned_dataframe' is missing, that's a
    real upstream bug (Cleaning Node didn't run, or wrote to the wrong
    key) and should fail loudly here, not quietly degrade to raw data.
    """
    df = state.get("cleaned_dataframe")
    if df is None:
        raise StatisticsNodeError(
            "'cleaned_dataframe' missing from state. Statistics Node must run "
            "after the Cleaning Node. If you intentionally renamed a state key, "
            "update GraphState and every node that reads it — do not add a "
            "silent fallback here."
        )
    return df


def _get_plan_analyses(state: dict) -> List[str]:
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        return list(plan.get("analyses", []) or [])
    return []


def _profile_columns(state: dict, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    profile = state.get("profile", {})
    numeric_cols = list(profile.get("numeric_columns", []) or [])
    categorical_cols = list(profile.get("categorical_columns", []) or [])
    datetime_cols = list(profile.get("datetime_columns", []) or [])

    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not categorical_cols:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not datetime_cols:
        datetime_cols = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()

    return numeric_cols, categorical_cols, datetime_cols


def _run_descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if not numeric_cols:
        return {"skipped_reason": "No numeric columns available."}

    stats = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "25%": float(series.quantile(0.25)),
            "50%": float(series.quantile(0.5)),
            "75%": float(series.quantile(0.75)),
            "max": float(series.max()),
            "sum": float(series.sum()),
        }

    return stats or {"skipped_reason": "No displayable numeric data."}


def _run_correlation_analysis(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if len(numeric_cols) < 2:
        return {"skipped_reason": "Fewer than 2 numeric columns."}

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(numeric_only=True)
    return corr.to_dict()


def _run_trend_analysis(df: pd.DataFrame, numeric_cols: list[str], datetime_cols: list[str]) -> dict:
    if not numeric_cols or not datetime_cols:
        return {"skipped_reason": "Requires at least one numeric and one datetime column."}

    dt_col = datetime_cols[0]
    ts_df = df.dropna(subset=[dt_col]).copy()
    ts_df[dt_col] = pd.to_datetime(ts_df[dt_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[dt_col]).sort_values(dt_col)

    if ts_df.empty:
        return {"skipped_reason": f"No valid datetime values found in {dt_col}."}

    result = {}
    for col in numeric_cols[:3]:
        series = pd.to_numeric(ts_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        first_value = float(series.iloc[0])
        last_value = float(series.iloc[-1])
        percent_change = ((last_value - first_value) / first_value * 100) if first_value not in (0.0, 0) else 0.0
        result[col] = {
            "date_column": dt_col,
            "first_value": first_value,
            "last_value": last_value,
            "percent_change": float(percent_change),
        }

    return result or {"skipped_reason": "No numeric trend series could be computed."}


def _run_outlier_detection(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if not numeric_cols:
        return {"skipped_reason": "No numeric columns available."}

    outliers = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        outlier_count = int(mask.sum())
        outliers[col] = {
            "outlier_count": outlier_count,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }

    return outliers or {"skipped_reason": "No outliers detected."}


def _run_value_counts_analysis(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    if not categorical_cols:
        return {"skipped_reason": "No categorical columns available."}

    result = {}
    for col in categorical_cols[:5]:
        counts = df[col].astype(str).value_counts().head(10)
        result[col] = counts.to_dict()

    return result or {"skipped_reason": "No displayable categorical data."}


def _run_grouped_summary_analysis(df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> dict:
    if not categorical_cols or not numeric_cols:
        return {"skipped_reason": "Requires at least one categorical and one numeric column."}

    primary_cat = categorical_cols[0]
    primary_num = numeric_cols[0]

    grouped = (
        df.groupby(primary_cat)[primary_num]
        .agg(["sum", "mean", "count", "std"])
        .sort_values("sum", ascending=False)
        .head(10)
    )
    grouped.columns = ["total", "average", "count", "std_dev"]
    grouped["rank"] = grouped["total"].rank(ascending=False, method="dense")

    return {
        "grouped_by": primary_cat,
        "measured": primary_num,
        "groups": grouped.to_dict(orient="index"),
    }


@node_error_boundary("statistics")
def statistics_node(state: dict) -> dict:
    # Fail loudly if canonical key 'cleaned_dataframe' is missing
    df = _get_dataframe(state)

    if df.empty:
        raise StatisticsNodeError("Statistics node requires a non-empty DataFrame in state.")

    plan_analyses = _get_plan_analyses(state)
    if not plan_analyses:
        raise StatisticsNodeError("No plan analyses found in state. Statistics Node must run after Planning Agent.")

    numeric_cols, categorical_cols, datetime_cols = _profile_columns(state, df)
    results: dict[str, Any] = {}
    unsupported_analyses: list[str] = []

    for analysis in plan_analyses:
        if analysis == "descriptive_stats":
            results[analysis] = _run_descriptive_stats(df, numeric_cols)
        elif analysis == "correlation_analysis":
            results[analysis] = _run_correlation_analysis(df, numeric_cols)
        elif analysis == "trend_analysis":
            results[analysis] = _run_trend_analysis(df, numeric_cols, datetime_cols)
        elif analysis == "outlier_detection":
            results[analysis] = _run_outlier_detection(df, numeric_cols)
        elif analysis == "value_counts_analysis":
            results[analysis] = _run_value_counts_analysis(df, categorical_cols)
        elif analysis == "grouped_summary_analysis":
            results[analysis] = _run_grouped_summary_analysis(df, categorical_cols, numeric_cols)
        else:
            unsupported_analyses.append(analysis)

    if not results:
        raise StatisticsNodeError("No supported analyses were requested by the plan.")

    logger.info(
        "Statistics Node completed — produced %s analyses (%s unsupported).",
        len(results),
        len(unsupported_analyses),
    )

    return {
        "statistics": {
            "results": results,
            "unsupported_analyses": unsupported_analyses,
        },
    }