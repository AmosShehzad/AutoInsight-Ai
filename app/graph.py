import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.schemas.state import GraphState
from app.nodes.file_loader import load_file_node
from app.nodes.validation import validate_data_node
from app.nodes.cleaning import clean_data_node
from app.nodes.profiling import profile_data_node
from app.nodes.planning_agent import planning_agent_node
from app.nodes.statistics import statistics_node
from app.nodes.visualization import visualization_node
from app.nodes.insight_agent import insight_agent_node
from app.nodes.critic_agent import critic_agent_node
from app.nodes.report_generator import report_generator_node   # NEW import

DB_PATH = "checkpoints.sqlite"


def _route_after_critic(state: GraphState) -> str:
    """
    Conditional edge router. Reads critic_feedback and decides:
    - approved -> move forward (report_generator)
    - not approved -> loop back to planning_agent, WITH revision_count incremented
    """
    feedback = state.get("critic_feedback", {})
    if feedback.get("approved", True):
        return "approved"
    return "needs_revision"


def _increment_revision_count(state: GraphState) -> dict:
    """
    Small pipeline node: bumps revision_count by 1 right before looping
    back to Planning. This is what MAX_REVISIONS in critic_agent.py checks
    against, so the safety valve actually has something real to count.
    """
    current = state.get("revision_count", 0)
    return {"revision_count": current + 1}


def build_graph(db_path: str = DB_PATH, interrupt_after: list[str] | None = None):
    graph = StateGraph(GraphState)

    graph.add_node("file_loader", load_file_node)
    graph.add_node("validation", validate_data_node)
    graph.add_node("cleaning", clean_data_node)
    graph.add_node("profiling", profile_data_node)
    graph.add_node("planning_agent", planning_agent_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insight_agent", insight_agent_node)
    graph.add_node("critic_agent", critic_agent_node)
    graph.add_node("increment_revision", _increment_revision_count)
    graph.add_node("report_generator", report_generator_node)   # NEW

    graph.set_entry_point("file_loader")

    graph.add_edge("file_loader", "validation")
    graph.add_edge("validation", "cleaning")
    graph.add_edge("cleaning", "profiling")
    graph.add_edge("profiling", "planning_agent")

    # Fan-out (Day 6)
    graph.add_edge("planning_agent", "statistics")
    graph.add_edge("planning_agent", "visualization")

    # Fan-in (Day 7)
    graph.add_edge("statistics", "insight_agent")
    graph.add_edge("visualization", "insight_agent")

    # Insight Agent -> Critic Agent
    graph.add_edge("insight_agent", "critic_agent")

    # THE CONDITIONAL EDGE — Reflection loop router
    graph.add_conditional_edges(
        "critic_agent",
        _route_after_critic,
        {
            "approved": "report_generator",   # CHANGED — now routes to report_generator
            "needs_revision": "increment_revision",
        },
    )

    graph.add_edge("increment_revision", "planning_agent")
    graph.add_edge("report_generator", END)   # NEW — final step of full pipeline

    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    checkpointer = SqliteSaver(conn, serde=serde)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


if __name__ == "__main__":
    app_graph = build_graph()
    config = {"configurable": {"thread_id": "demo-run-day9"}}
    result = app_graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config,
    )
    print("Revision count:", result.get("revision_count", 0))
    print("Critic feedback:", result.get("critic_feedback"))
    print("\nReport generated at:", result.get("report_path"))