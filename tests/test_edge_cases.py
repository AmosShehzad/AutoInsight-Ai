import pandas as pd
import pytest
from app.graph import build_graph

requires_api_key = pytest.mark.skipif(
    __import__("os").getenv("GROQ_API_KEY") is None,
    reason="GROQ_API_KEY not set"
)


@pytest.fixture
def graph():
    return build_graph(db_path=":memory:")


@requires_api_key
def test_one_row_dataset_does_not_crash(graph, tmp_path):
    file_path = tmp_path / "one_row.csv"
    pd.DataFrame({"a": [1], "b": ["x"]}).to_csv(file_path, index=False)
    result = graph.invoke({"file_path": str(file_path)}, config={"configurable": {"thread_id": "edge-onerow"}})
    assert "report_path" in result


@requires_api_key
def test_all_text_dataset_does_not_crash(graph, tmp_path):
    file_path = tmp_path / "all_text.csv"
    pd.DataFrame({"name": ["A", "B", "C"], "note": ["x", "y", "z"]}).to_csv(file_path, index=False)
    result = graph.invoke({"file_path": str(file_path)}, config={"configurable": {"thread_id": "edge-alltext"}})
    assert "report_path" in result


def test_empty_dataset_raises_clean_error(tmp_path):
    from app.nodes.file_loader import FileLoadError
    file_path = tmp_path / "empty.csv"
    file_path.write_text("a,b,c\n")
    from app.graph import build_graph
    g = build_graph(db_path=":memory:")
    with pytest.raises(FileLoadError):
        g.invoke({"file_path": str(file_path)}, config={"configurable": {"thread_id": "edge-empty"}})