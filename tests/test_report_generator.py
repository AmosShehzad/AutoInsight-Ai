import os
import pandas as pd
import pytest
from app.nodes.report_generator import report_generator_node, ReportGeneratorError


def _full_state(with_chart=False, tmp_path=None):
    """Builds a minimal but complete state dict covering everything Report Generator needs."""
    chart_file = None
    if with_chart:
        # create a tiny real PNG so Image() doesn't fail on a fake path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        chart_file = str(tmp_path / "test_chart.png")
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.savefig(chart_file)
        plt.close()

    return {
        "profile": {
            "row_count": 10, "column_count": 3,
            "numeric_columns": ["a"], "categorical_columns": ["b"], "datetime_columns": [],
        },
        "validation_report": {"health_score": 95.0, "missing_cells": 1, "duplicate_rows": 0},
        "cleaning_report": {"duplicates_removed": 0, "rows_after_cleaning": 10},
        "plan": {"summary": "Test plan summary.", "analyses": ["descriptive_stats"], "charts": []},
        "statistics": {"results": {"descriptive_stats": {"a": {"mean": 5.0}}}},
        "visualizations": {
            "generated": (
                [{"chart_type": "line", "columns": ["a"], "reason": "test chart", "file_path": chart_file}]
                if with_chart else []
            )
        },
        "insights": ["The average of column a is 5.0."],
    }


def _rich_state(tmp_path=None):
    state = _full_state(tmp_path=tmp_path)
    state["profile"] = {
        "row_count": 250,
        "column_count": 5,
        "numeric_columns": ["revenue", "profit"],
        "categorical_columns": ["region"],
        "datetime_columns": ["date"],
    }
    state["validation_report"] = {
        "health_score": 97.5,
        "missing_cells": 0,
        "duplicate_rows": 1,
    }
    state["cleaning_report"] = {
        "duplicates_removed": 1,
        "rows_after_cleaning": 249,
    }
    state["plan"] = {
        "summary": "Focus on revenue growth, margin quality, and regional concentration.",
        "analyses": ["trend_analysis", "correlation_analysis", "outlier_detection", "descriptive_stats"],
        "charts": [],
    }
    state["statistics"] = {
        "results": {
            "trend_analysis": {
                "revenue": {"date_column": "date", "first_value": 100.0, "last_value": 145.0, "percent_change": 45.0}
            },
            "correlation_analysis": {
                "revenue": {"profit": 0.83},
                "profit": {"revenue": 0.83},
            },
            "outlier_detection": {
                "revenue": {"outlier_count": 2}
            },
            "descriptive_stats": {
                "revenue": {"mean": 128.2, "sum": 32050.0}
            },
        }
    }
    state["insights"] = [
        "Revenue increased by 45.0% over the observed period.",
        "Revenue and profit moved together with strong positive correlation.",
    ]
    return state


def test_report_is_generated_successfully(tmp_path, monkeypatch):
    # redirect REPORTS_DIR to a temp folder so tests don't pollute outputs/reports
    import app.nodes.report_generator as rg_module
    monkeypatch.setattr(rg_module, "REPORTS_DIR", str(tmp_path))

    state = _full_state()
    result = report_generator_node(state)

    assert "report_path" in result
    assert os.path.exists(result["report_path"])
    assert result["report_path"].endswith(".pdf")
    assert os.path.getsize(result["report_path"]) > 0


def test_report_includes_chart_image(tmp_path, monkeypatch):
    import app.nodes.report_generator as rg_module
    monkeypatch.setattr(rg_module, "REPORTS_DIR", str(tmp_path))

    state = _full_state(with_chart=True, tmp_path=tmp_path)
    result = report_generator_node(state)

    assert os.path.exists(result["report_path"])
    # A PDF with an embedded image should be meaningfully larger than a text-only one
    text_only_state = _full_state()
    text_only_result = report_generator_node(text_only_state)
    assert os.path.getsize(result["report_path"]) > os.path.getsize(text_only_result["report_path"])


def test_rich_report_builds_successfully(tmp_path, monkeypatch):
    import app.nodes.report_generator as rg_module
    monkeypatch.setattr(rg_module, "REPORTS_DIR", str(tmp_path))

    result = report_generator_node(_rich_state(tmp_path=tmp_path))

    assert os.path.exists(result["report_path"])
    assert os.path.getsize(result["report_path"]) > 0


def test_missing_required_fields_raises_error():
    state = {"profile": {"row_count": 5}}  # everything else missing
    with pytest.raises(ReportGeneratorError):
        report_generator_node(state)


def test_missing_chart_file_does_not_crash_report(tmp_path, monkeypatch):
    """
    Defensive test: if a chart's file_path in state points to a file that
    doesn't actually exist on disk, the report should still generate
    (with a placeholder note), not crash entirely.
    """
    import app.nodes.report_generator as rg_module
    monkeypatch.setattr(rg_module, "REPORTS_DIR", str(tmp_path))

    state = _full_state()
    state["visualizations"] = {
        "generated": [{"chart_type": "line", "columns": ["a"], "reason": "test",
                       "file_path": "generated_charts/does_not_exist.png"}]
    }

    result = report_generator_node(state)
    assert os.path.exists(result["report_path"])  # report still built successfully