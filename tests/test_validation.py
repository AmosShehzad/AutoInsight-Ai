import pandas as pd
from app.nodes.validation import validate_data_node


def test_validation_on_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    state = {"dataframe": df}

    result = validate_data_node(state)
    report = result["validation_report"]

    assert report["row_count"] == 3
    assert report["missing_cells"] == 0
    assert report["duplicate_rows"] == 0
    assert report["health_score"] == 100.0


def test_validation_on_messy_data():
    df = pd.DataFrame({"a": [1, None, 1], "b": [4, 5, 4]})
    # row 0 and row 2 are duplicates, and there's a missing value
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    state = {"dataframe": df}
    result = validate_data_node(state)
    report = result["validation_report"]

    assert report["missing_cells"] >= 1
    assert report["duplicate_rows"] >= 1
    assert report["health_score"] < 100.0