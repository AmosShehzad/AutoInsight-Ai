import os
import pytest
from app.nodes.critic_agent import critic_agent_node, MAX_REVISIONS, CriticAgentError

requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests"
)


def _minimal_state(revision_count=0):
    return {
        "plan": {"analyses": ["descriptive_stats"], "charts": []},
        "statistics": {"results": {"descriptive_stats": {"a": {"mean": 5.0}}},
                        "requested_analyses": ["descriptive_stats"], "unsupported_analyses": []},
        "visualizations": {"generated": [], "requested_charts": [], "unsupported_charts": []},
        "insights": ["The average of column a is 5.0."],
        "revision_count": revision_count,
    }


def test_safety_valve_forces_approval_at_max_revisions():
    """
    Does NOT need an API key — this tests the safety valve, which
    short-circuits BEFORE any LLM call happens.
    """
    state = _minimal_state(revision_count=MAX_REVISIONS)
    result = critic_agent_node(state)

    assert result["critic_feedback"]["approved"] is True
    assert "max revision limit" in result["critic_feedback"]["reason"].lower()


def test_missing_dependencies_raises_error():
    state = {"plan": {"analyses": []}}  # missing statistics, visualizations, insights
    with pytest.raises(CriticAgentError):
        critic_agent_node(state)


@requires_api_key
def test_critic_produces_valid_feedback_structure():
    state = _minimal_state()
    result = critic_agent_node(state)
    feedback = result["critic_feedback"]

    assert "approved" in feedback
    assert isinstance(feedback["approved"], bool)
    assert "reason" in feedback
    assert len(feedback["reason"]) > 0