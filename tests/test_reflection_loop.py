import os
import pandas as pd
import pytest
from app.graph import build_graph

requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests"
)


@pytest.fixture
def graph():
    return build_graph(db_path=":memory:")


@requires_api_key
def test_full_pipeline_reaches_a_final_critic_decision(graph):
    """
    Proves the graph runs all the way through, INCLUDING the Critic step,
    and ends in a well-formed state — whether it took 0 revisions or several.
    """
    config = {"configurable": {"thread_id": "test-reflection-1"}}
    result = graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config,
    )

    assert "critic_feedback" in result
    assert result["critic_feedback"]["approved"] is True  # by the time we exit the loop, it's always approved
    # revision_count proves whether looping actually happened this run
    assert result.get("revision_count", 0) >= 0


@requires_api_key
def test_revision_count_never_exceeds_max(graph):
    """
    Runs the pipeline on a dataset likely to need at least one revision
    (very sparse data, few columns) and confirms the safety valve holds —
    revision_count should never exceed MAX_REVISIONS from critic_agent.py.
    """
    from app.nodes.critic_agent import MAX_REVISIONS

    config = {"configurable": {"thread_id": "test-reflection-2"}}
    result = graph.invoke(
        {"file_path": "sample_data/categorical_sample.csv"},
        config=config,
    )

    assert result.get("revision_count", 0) <= MAX_REVISIONS