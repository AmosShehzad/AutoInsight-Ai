import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from app.nodes.statistics import statistics_node, StatisticsNodeError


def _fake_profile(df: pd.DataFrame) -> dict:
    """Small helper to build a minimal profile dict for testing, without needing the real Profiling Node."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": [],
        "numeric_stats": {
            col: {
                "mean": float(df[col].mean()), "median": float(df[col].median()),
                "std": float(df[col].std()) if len(df) > 1 else 0.0,
                "min": float(df[col].min()), "max": float(df[col].max()),
                "unique_values": int(df[col].nunique()),
            } for col in numeric_cols
        },
        "categorical_stats": {
            col: {"unique_values": int(df[col].nunique()),
                  "top_5_values": df[col].value_counts().head(5).to_dict()}
            for col in categorical_cols
        },
    }


def test_descriptive_stats_and_correlation_run_successfully():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    profile = _fake_profile(df)
    state = {
        "cleaned_dataframe": df,
        "profile": profile,
        "plan": {"analyses": ["descriptive_stats", "correlation_analysis"]},
    }

    result = statistics_node(state)
    stats = result["statistics"]

    assert "descriptive_stats" in stats["results"]
    assert "correlation_analysis" in stats["results"]
    assert stats["unsupported_analyses"] == []


def test_correlation_analysis_skips_gracefully_with_one_numeric_column():
    df = pd.DataFrame({"a": [1, 2, 3], "category": ["x", "y", "z"]})
    profile = _fake_profile(df)
    state = {
        "cleaned_dataframe": df,
        "profile": profile,
        "plan": {"analyses": ["correlation_analysis"]},
    }

    result = statistics_node(state)
    corr_result = result["statistics"]["results"]["correlation_analysis"]

    assert "skipped_reason" in corr_result


def test_unsupported_analysis_name_is_recorded_not_crashed():
    """
    THE key test for today: proves the fallback logic works when the
    Planning Agent requests something the Statistics Node doesn't support.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    profile = _fake_profile(df)
    state = {
        "cleaned_dataframe": df,
        "profile": profile,
        "plan": {"analyses": ["descriptive_stats", "quantum_flux_analysis"]},
    }

    result = statistics_node(state)
    stats = result["statistics"]

    assert "descriptive_stats" in stats["results"]
    assert "quantum_flux_analysis" in stats["unsupported_analyses"]


def test_missing_plan_raises_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    profile = _fake_profile(df)
    state = {"cleaned_dataframe": df, "profile": profile}

    with pytest.raises(StatisticsNodeError):
        statistics_node(state)


def test_trend_analysis_computes_percent_change():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "revenue": [100, 150, 200],
    })
    profile = {
        "numeric_columns": ["revenue"],
        "categorical_columns": [],
        "datetime_columns": ["date"],
        "numeric_stats": {}, "categorical_stats": {},
    }
    state = {
        "cleaned_dataframe": df,
        "profile": profile,
        "plan": {"analyses": ["trend_analysis"]},
    }

    result = statistics_node(state)
    trend = result["statistics"]["results"]["trend_analysis"]["revenue"]

    assert trend["first_value"] == 100.0
    assert trend["last_value"] == 200.0
    assert trend["percent_change"] == 100.0