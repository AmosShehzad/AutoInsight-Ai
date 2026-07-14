import pandas as pd
import pytest
from app.graph import build_graph
from app.nodes.file_loader import FileLoadError


@pytest.fixture
def graph():
    return build_graph()


def test_clean_dataset_end_to_end(graph, tmp_path):
    file_path = tmp_path / "clean.csv"
    pd.DataFrame({
        "name": ["Ali", "Sara", "Bilal"],
        "age": [28, 34, 25],
        "salary": [75000, 92000, 68000],
    }).to_csv(file_path, index=False)

    result = graph.invoke(
        {"file_path": str(file_path)},
        config={"configurable": {"thread_id": "test-e2e-clean"}},
    )

    assert result["validation_report"]["health_score"] == 100.0
    assert result["cleaning_report"]["duplicates_removed"] == 0
    assert result["profile"]["row_count"] == 3
    assert "age" in result["profile"]["numeric_columns"]
    assert "name" in result["profile"]["categorical_columns"]


def test_messy_dataset_end_to_end(graph, tmp_path):
    file_path = tmp_path / "messy.csv"
    df = pd.DataFrame({
        "name": ["Ali", "Sara", "Ali"],
        "age": [28, None, 28],
        "salary": [75000, 92000, 75000],
    })
    df.to_csv(file_path, index=False)

    result = graph.invoke(
        {"file_path": str(file_path)},
        config={"configurable": {"thread_id": "test-e2e-messy"}},
    )

    # validation should catch the issues before cleaning fixes them
    assert result["validation_report"]["missing_cells"] >= 1
    assert result["validation_report"]["duplicate_rows"] >= 1

    # cleaning should have resolved them
    assert result["cleaning_report"]["duplicates_removed"] >= 1
    assert result["cleaning_report"]["rows_after_cleaning"] < result["cleaning_report"]["original_rows"]
    assert "age" in result["cleaning_report"]["missing_value_handling"]

    # profile should be built on the CLEANED data (no more missing values)
    cleaned_df = result["cleaned_dataframe"]
    assert cleaned_df["age"].isnull().sum() == 0


def test_empty_dataset_raises_error(graph, tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("name,age,city,salary\n")

    with pytest.raises(FileLoadError):
        graph.invoke(
            {"file_path": str(file_path)},
            config={"configurable": {"thread_id": "test-e2e-empty"}},
        )