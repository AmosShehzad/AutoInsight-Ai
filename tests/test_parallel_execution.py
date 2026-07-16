import pandas as pd
import pytest
from app.graph import build_graph


@pytest.fixture
def graph():
    return build_graph(db_path=":memory:")


def test_statistics_and_visualization_both_complete(graph, tmp_path):
    """
    Proves fan-out/fan-in works end-to-end: after Planning Agent runs,
    BOTH statistics and visualizations should exist in the final state —
    proving both parallel branches ran and both merged back correctly.
    """
    file_path = tmp_path / "data.csv"
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "revenue": [100, 120, 90, 150, 170, 200],
    }).to_csv(file_path, index=False)

    result = graph.invoke(
        {"file_path": str(file_path)},
        config={"configurable": {"thread_id": "test-parallel-1"}},
    )

    # Both branches must have run and written their own state keys —
    # this is the proof that fan-out/fan-in merged state correctly,
    # with no conflicts between the two parallel nodes.
    assert "statistics" in result
    assert "visualizations" in result
    assert "results" in result["statistics"]
    assert "generated" in result["visualizations"]