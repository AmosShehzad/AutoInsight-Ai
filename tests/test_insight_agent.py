import os
import re
import pytest
from app.graph import build_graph

requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests"
)


@pytest.fixture
def graph():
    return build_graph(db_path=":memory:")


def _contains_a_number(text: str) -> bool:
    """Simple check: does this sentence contain at least one digit? Used to
    verify insights are grounded in real numbers, not generic filler."""
    return bool(re.search(r"\d", text))


@requires_api_key
def test_insights_are_generated_and_grounded_in_numbers(graph):
    config = {"configurable": {"thread_id": "test-insight-ts"}}
    result = graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config,
    )

    insights = result["insights"]
    assert 3 <= len(insights) <= 7

    # EVERY insight should reference a real number — the key requirement for today
    ungrounded = [i for i in insights if not _contains_a_number(i)]
    assert not ungrounded, f"Found insights with no numbers (generic filler): {ungrounded}"


@requires_api_key
def test_insights_differ_across_three_different_datasets(graph):
    """
    THE key test for today: proves insight quality/content genuinely
    changes based on what's actually in the data across 3 distinct datasets.
    """
    datasets = {
        "timeseries": "sample_data/timeseries_sample.csv",
        "categorical": "sample_data/categorical_sample.csv",
        "outlier": "sample_data/outlier_sample.csv",
    }

    all_insights = {}
    for name, path in datasets.items():
        config = {"configurable": {"thread_id": f"test-insight-{name}"}}
        result = graph.invoke({"file_path": path}, config=config)
        all_insights[name] = " ".join(result["insights"]).lower()

    # No two datasets should produce identical insight text —
    # if they do, the agent isn't actually reasoning about the specific numbers.
    assert all_insights["timeseries"] != all_insights["categorical"]
    assert all_insights["timeseries"] != all_insights["outlier"]
    assert all_insights["categorical"] != all_insights["outlier"]


@requires_api_key
def test_insights_detailed_includes_source_analysis(graph):
    config = {"configurable": {"thread_id": "test-insight-detailed"}}
    result = graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config,
    )

    detailed = result["insights_detailed"]
    assert len(detailed) > 0
    for item in detailed:
        assert "text" in item
        assert "source_analysis" in item
        assert item["source_analysis"] != ""