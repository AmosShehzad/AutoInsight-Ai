import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.logger import get_logger
from app.schemas.state import GraphState

logger = get_logger(__name__)


class StatisticsNodeError(Exception):
    """Raised when the Statistics Node cannot execute with the provided state."""
    pass


def run_comprehensive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes an analytics suite leveraging 21 core Pandas functions
    to generate deep business metrics and chart-ready payloads.
    """
    results = {}
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    # Fallback attempt to parse datetime if not explicitly typed
    if not datetime_cols:
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols.append(col)
                    break
                except Exception:
                    pass

    # ---------------------------------------------------------------------------
    # 1. describe(), quantile(), nunique()
    # ---------------------------------------------------------------------------
    results["cardinality"] = {col: df[col].nunique() for col in df.columns} # nunique()
    
    if numeric_cols:
        num_df = df[numeric_cols]
        desc = num_df.describe().to_dict() # describe()
        
        # Add quantiles
        for col in numeric_cols:
            q_vals = {
                "q25": float(df[col].quantile(0.25)), # quantile()
                "q50": float(df[col].quantile(0.50)),
                "q75": float(df[col].quantile(0.75)),
                "q90": float(df[col].quantile(0.90)),
                "q99": float(df[col].quantile(0.99))
            }
            if col in desc:
                desc[col].update(q_vals)
        results["descriptive_stats"] = desc

    # ---------------------------------------------------------------------------
    # 2. corr()
    # ---------------------------------------------------------------------------
    if len(numeric_cols) >= 2:
        results["correlation_analysis"] = df[numeric_cols].corr().to_dict() # corr()
    else:
        results["correlation_analysis"] = {"skipped_reason": "Fewer than 2 numeric columns."}

    # ---------------------------------------------------------------------------
    # 3. value_counts(), nlargest(), nsmallest()
    # ---------------------------------------------------------------------------
    val_counts = {}
    for col in categorical_cols[:5]: # Top 5 categorical dimensions
        vc = df[col].value_counts() # value_counts()
        val_counts[col] = {
            "top_5": vc.nlargest(5).to_dict(), # nlargest()
            "bottom_5": vc.nsmallest(5).to_dict() # nsmallest()
        }
    results["categorical_breakdown"] = val_counts

    # ---------------------------------------------------------------------------
    # 4. groupby(), agg(), rank(), transform(), merge()
    # ---------------------------------------------------------------------------
    if categorical_cols and numeric_cols:
        primary_cat = categorical_cols[0]
        primary_num = numeric_cols[0]
        
        # groupby() + agg()
        grouped = df.groupby(primary_cat).agg({ # groupby(), agg()
            primary_num: ['sum', 'mean', 'count', 'std']
        })
        grouped.columns = ['total', 'average', 'count', 'std_dev']
        
        # rank()
        grouped['rank'] = grouped['total'].rank(ascending=False, method='dense') # rank()
        
        # transform() + merge()
        df_copy = df.copy()
        df_copy['cat_total'] = df_copy.groupby(primary_cat)[primary_num].transform('sum') # transform()
        df_copy['pct_of_total'] = (df_copy[primary_num] / df_copy[primary_num].sum()) * 100
        
        results["group_performance"] = {
            "grouped_by": primary_cat,
            "measured": primary_num,
            "groups": grouped.head(10).to_dict(orient="index")
        }

    # ---------------------------------------------------------------------------
    # 5. pivot_table(), melt(), crosstab()
    # ---------------------------------------------------------------------------
    if len(categorical_cols) >= 2 and numeric_cols:
        cat1, cat2 = categorical_cols[0], categorical_cols[1]
        primary_num = numeric_cols[0]
        
        # pivot_table()
        pivot = pd.pivot_table( # pivot_table()
            df, values=primary_num, index=cat1, columns=cat2, aggfunc='sum', fill_value=0
        )
        
        # melt()
        pivot_reset = pivot.reset_index()
        melted = pd.melt(pivot_reset, id_vars=[cat1], var_name=cat2, value_name='total_value') # melt()
        
        # crosstab()
        ct = pd.crosstab(df[cat1], df[cat2], normalize='index') * 100 # crosstab()
        
        results["pivot_matrix"] = {
            "dimensions": [cat1, cat2],
            "metric": primary_num,
            "pivot_data": pivot.head(8).to_dict(),
            "crosstab_pct": ct.head(8).to_dict()
        }

    # ---------------------------------------------------------------------------
    # 6. Time-Series: resample(), pct_change(), rolling(), shift(), diff()
    # ---------------------------------------------------------------------------
    if datetime_cols and numeric_cols:
        dt_col = datetime_cols[0]
        primary_num = numeric_cols[0]
        
        ts_df = df.set_index(dt_col).sort_index()
        
        # resample()
        monthly = ts_df[primary_num].resample('ME').sum().to_frame() # resample()
        
        # pct_change(), shift(), diff(), rolling()
        monthly['pct_change'] = monthly[primary_num].pct_change() * 100 # pct_change()
        monthly['prev_period'] = monthly[primary_num].shift(1) # shift()
        monthly['absolute_diff'] = monthly[primary_num].diff() # diff()
        monthly['rolling_3m_avg'] = monthly[primary_num].rolling(window=3).mean() # rolling()
        
        results["time_series_trend"] = {
            "date_column": dt_col,
            "metric": primary_num,
            "monthly_summary": monthly.tail(12).to_dict(orient="index"),
            "total_percent_change": float(
                ((monthly[primary_num].iloc[-1] - monthly[primary_num].iloc[0]) / monthly[primary_num].iloc[0]) * 100
            ) if len(monthly) > 1 else 0.0
        }

    # ---------------------------------------------------------------------------
    # 7. explode() handling (for tag/array columns if present)
    # ---------------------------------------------------------------------------
    for col in categorical_cols:
        if df[col].astype(str).str.contains(',').any():
            exploded_df = df.assign(**{col: df[col].astype(str).str.split(',')}).explode(col) # explode()
            results["exploded_tags"] = {
                "column": col,
                "top_tags": exploded_df[col].str.strip().value_counts().head(5).to_dict()
            }
            break

    return results


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

def statistics_node(state: dict) -> dict:
    # Check all possible keys used across your nodes
    df = state.get("cleaned_dataframe")
    if df is None:
        df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")
    if df is None:
        df = state.get("raw_data")

    if df is None or df.empty:
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