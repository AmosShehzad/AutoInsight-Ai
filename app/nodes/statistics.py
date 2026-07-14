import pandas as pd
from app.schemas.state import GraphState


class StatisticsNodeError(Exception):
    """Raised only for unrecoverable errors — e.g. no data to work with at all."""
    pass


# ---------------------------------------------------------------------------
# Each function below performs ONE type of analysis. They all take the same
# two inputs (dataframe + profile) and return a plain dict of results.
# Small, single-purpose functions = independently testable, easy to extend.
# ---------------------------------------------------------------------------

def _descriptive_stats(df: pd.DataFrame, profile: dict) -> dict:
    """Mean, median, std, min, max for every numeric column — reuses profile if present."""
    numeric_cols = profile.get("numeric_columns", [])
    if not numeric_cols:
        return {"skipped_reason": "No numeric columns available for descriptive stats."}
    return {col: profile["numeric_stats"][col] for col in numeric_cols}


def _correlation_analysis(df: pd.DataFrame, profile: dict) -> dict:
    """Pearson correlation matrix between all numeric columns."""
    numeric_cols = profile.get("numeric_columns", [])
    if len(numeric_cols) < 2:
        return {"skipped_reason": "Correlation analysis needs at least 2 numeric columns."}

    corr_matrix = df[numeric_cols].corr(numeric_only=True).round(3)
    return {col: corr_matrix[col].to_dict() for col in corr_matrix.columns}


def _trend_analysis(df: pd.DataFrame, profile: dict) -> dict:
    """Basic trend detection: sorts by a datetime column and computes overall % change."""
    datetime_cols = profile.get("datetime_columns", [])
    numeric_cols = profile.get("numeric_columns", [])

    if not datetime_cols:
        return {"skipped_reason": "Trend analysis needs at least 1 datetime column."}
    if not numeric_cols:
        return {"skipped_reason": "Trend analysis needs at least 1 numeric column to track."}

    date_col = datetime_cols[0]
    sorted_df = df.sort_values(by=date_col)

    trends = {}
    for col in numeric_cols:
        first_val = float(sorted_df[col].iloc[0])
        last_val = float(sorted_df[col].iloc[-1])
        percent_change = round(((last_val - first_val) / first_val) * 100, 2) if first_val != 0 else None
        trends[col] = {
            "first_value": first_val,
            "last_value": last_val,
            "percent_change": percent_change,
        }
    return trends


def _outlier_detection(df: pd.DataFrame, profile: dict) -> dict:
    """Simple IQR-based outlier detection for each numeric column."""
    numeric_cols = profile.get("numeric_columns", [])
    if not numeric_cols:
        return {"skipped_reason": "Outlier detection needs at least 1 numeric column."}

    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_values = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()

        outliers[col] = {
            "outlier_count": len(outlier_values),
            "outlier_values": [float(v) for v in outlier_values[:10]],
        }
    return outliers


def _value_counts_analysis(df: pd.DataFrame, profile: dict) -> dict:
    """Value counts for categorical columns — reuses profile if present."""
    categorical_cols = profile.get("categorical_columns", [])
    if not categorical_cols:
        return {"skipped_reason": "No categorical columns available for value counts."}
    return {col: profile["categorical_stats"][col]["top_5_values"] for col in categorical_cols}


# ---------------------------------------------------------------------------
# THE DISPATCH TABLE: maps a plan's analysis NAME (string) to the function
# that performs it. Core pattern for today — look up by name, call it.
# ---------------------------------------------------------------------------
ANALYSIS_DISPATCH = {
    "descriptive_stats": _descriptive_stats,
    "correlation_analysis": _correlation_analysis,
    "trend_analysis": _trend_analysis,
    "outlier_detection": _outlier_detection,
    "value_counts_analysis": _value_counts_analysis,
}


def statistics_node(state: GraphState) -> GraphState:
    """
    LangGraph node (PIPELINE NODE — deterministic, no LLM call).
    Reads state['plan']['analyses'] and executes each requested analysis
    using the dispatch table above. Unsupported names are recorded, not crashed on.
    """
    df = state.get("cleaned_dataframe")
    profile = state.get("profile")
    plan = state.get("plan")

    if df is None:
        raise StatisticsNodeError("No 'cleaned_dataframe' in state. Statistics Node needs cleaned data.")
    if not profile:
        raise StatisticsNodeError("No 'profile' in state. Statistics Node needs the dataset profile.")
    if not plan:
        raise StatisticsNodeError("No 'plan' in state. Statistics Node must run AFTER the Planning Agent.")

    requested_analyses = plan.get("analyses", [])
    results = {}
    unsupported = []

    for analysis_name in requested_analyses:
        analysis_fn = ANALYSIS_DISPATCH.get(analysis_name)

        if analysis_fn is None:
            # FALLBACK LOGIC: Planning Agent asked for something unsupported.
            # Record it clearly instead of crashing the pipeline.
            unsupported.append(analysis_name)
            continue

        try:
            results[analysis_name] = analysis_fn(df, profile)
        except Exception as e:
            # Even a supported analysis can fail unexpectedly on weird data —
            # record the failure per-analysis, don't kill the whole node.
            results[analysis_name] = {"error": f"Analysis failed: {e}"}

    statistics_output = {
        "results": results,
        "requested_analyses": requested_analyses,
        "unsupported_analyses": unsupported,
    }

    return {**state, "statistics": statistics_output}