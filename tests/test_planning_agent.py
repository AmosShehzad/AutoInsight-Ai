import os
import pytest
from app.graph import build_graph

# Skip these tests automatically if no API key is set (e.g. in CI environments)
requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests"
)


@pytest.fixture
def graph():
    return build_graph(db_path=":memory:")  # in-memory DB, don't pollute real checkpoints


@requires_api_key
def test_planning_agent_produces_valid_plan_structure(graph):
    config = {"configurable": {"thread_id": "test-plan-structure"}}
    result = graph.invoke(
        {"file_path": "sample_data/clean_sample.csv"},
        config=config,
    )

    plan = result["plan"]
    assert "analyses" in plan
    assert "charts" in plan
    assert "summary" in plan
    assert isinstance(plan["analyses"], list)
    assert isinstance(plan["charts"], list)
    assert len(plan["analyses"]) > 0


@requires_api_key
def test_plan_differs_between_timeseries_and_categorical_data(graph):
    """
    THE key test for today: proves the agent is actually reasoning,
    not returning a fixed/generic plan regardless of input.
    """
    config_ts = {"configurable": {"thread_id": "test-plan-timeseries"}}
    result_ts = graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config_ts,
    )

    config_cat = {"configurable": {"thread_id": "test-plan-categorical"}}
    result_cat = graph.invoke(
        {"file_path": "sample_data/categorical_sample.csv"},
        config=config_cat,
    )

    plan_ts = result_ts["plan"]
    plan_cat = result_cat["plan"]

    # The two plans' chosen analyses should NOT be identical —
    # if they are, the agent is ignoring the actual data.
    assert plan_ts["analyses"] != plan_cat["analyses"] or \
           plan_ts["charts"] != plan_cat["charts"], \
        "Plans are identical for very different datasets — agent may not be reasoning about input"

    # Time-series data should reasonably mention trend-related analysis
    ts_analyses_text = " ".join(plan_ts["analyses"]).lower()
    assert "trend" in ts_analyses_text or "time" in ts_analyses_text, \
        "Time-series dataset didn't produce any trend/time-related analysis"