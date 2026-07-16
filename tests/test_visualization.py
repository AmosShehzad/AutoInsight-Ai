import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from app.nodes.visualization import visualization_node, VisualizationNodeError, CHARTS_DIR


def test_bar_chart_is_generated(tmp_path):
    df = pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
    state = {
        "cleaned_dataframe": df,
        "plan": {"charts": [{"chart_type": "bar", "columns": ["category"], "reason": "test"}]},
    }

    result = visualization_node(state)
    viz = result["visualizations"]

    assert len(viz["generated"]) == 1
    assert viz["generated"][0]["chart_type"] == "bar"
    assert os.path.exists(viz["generated"][0]["file_path"])


def test_line_chart_is_generated():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "revenue": [100, 150, 200],
    })
    state = {
        "cleaned_dataframe": df,
        "plan": {"charts": [{"chart_type": "line", "columns": ["date", "revenue"], "reason": "trend"}]},
    }

    result = visualization_node(state)
    assert len(result["visualizations"]["generated"]) == 1
    assert os.path.exists(result["visualizations"]["generated"][0]["file_path"])


def test_unsupported_chart_type_is_recorded_not_crashed():
    """
    THE key fallback test: proves an invalid chart_type doesn't crash
    the node, and valid charts alongside it still get generated.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "category": ["x", "y", "z"]})
    state = {
        "cleaned_dataframe": df,
        "plan": {"charts": [
            {"chart_type": "bar", "columns": ["category"], "reason": "test"},
            {"chart_type": "pie_3d_exploded", "columns": ["a"], "reason": "made up type"},
        ]},
    }

    result = visualization_node(state)
    viz = result["visualizations"]

    assert len(viz["generated"]) == 1
    assert viz["generated"][0]["chart_type"] == "bar"
    assert "pie_3d_exploded" in viz["unsupported_charts"]


def test_missing_column_is_recorded_not_crashed():
    """Proves a hallucinated column name is caught before matplotlib ever tries to use it."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    state = {
        "cleaned_dataframe": df,
        "plan": {"charts": [{"chart_type": "histogram", "columns": ["column_that_does_not_exist"], "reason": "test"}]},
    }

    result = visualization_node(state)
    viz = result["visualizations"]

    assert len(viz["generated"]) == 0
    assert len(viz["unsupported_charts"]) == 1


def test_missing_plan_raises_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    state = {"cleaned_dataframe": df}

    with pytest.raises(VisualizationNodeError):
        visualization_node(state)