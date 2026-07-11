import os
import pandas as pd
import pytest
from app.nodes.file_loader import load_file_node, FileLoadError


def test_load_valid_csv(tmp_path):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file_path, index=False)

    state = {"file_path": str(file_path)}
    result = load_file_node(state)

    assert "dataframe" in result
    assert result["dataframe"].shape == (2, 2)


def test_missing_file_raises_error():
    state = {"file_path": "does_not_exist.csv"}
    with pytest.raises(FileLoadError):
        load_file_node(state)


def test_unsupported_extension_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    state = {"file_path": str(file_path)}
    with pytest.raises(FileLoadError):
        load_file_node(state)


def test_empty_csv_raises_error(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    state = {"file_path": str(file_path)}
    with pytest.raises(FileLoadError):
        load_file_node(state)