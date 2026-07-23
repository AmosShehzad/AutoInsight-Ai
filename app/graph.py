import sys
import uuid
import sqlite3
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from app.nodes.report_generator import report_generator_node

DB_PATH = "checkpoints.sqlite"


class PipelineExecutionError(Exception):
    """User-facing wrapper for any pipeline failure — carries a clean message."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


def run_pipeline_safely(app_graph, initial_state: dict, config: dict) -> dict:
    """
    Wraps graph.invoke() with a friendly error boundary. Instead of a raw
    traceback reaching the end user, returns a clear stage + message.
    Use this in FastAPI (Day 11) instead of calling .invoke() directly.
    """
    try:
        return app_graph.invoke(initial_state, config=config)
    except Exception as e:
        stage = type(e).__module__.split(".")[-1] if hasattr(type(e), "__module__") else "unknown"
        raise PipelineExecutionError(stage=stage, message=str(e))


def _route_after_critic(state: GraphState) -> str:
    feedback = state.get("critic_feedback", {})
    if feedback.get("approved", True):
        return "approved"
    return "needs_revision"


def _increment_revision_count(state: GraphState) -> dict:
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
    graph.add_node("report_generator", report_generator_node)

    graph.set_entry_point("file_loader")

    graph.add_edge("file_loader", "validation")
    graph.add_edge("validation", "cleaning")
    graph.add_edge("cleaning", "profiling")
    graph.add_edge("profiling", "planning_agent")

    # Fan-out
    graph.add_edge("planning_agent", "statistics")
    graph.add_edge("planning_agent", "visualization")

    # Fan-in
    graph.add_edge("statistics", "insight_agent")
    graph.add_edge("visualization", "insight_agent")

    # Insight Agent -> Critic Agent
    graph.add_edge("insight_agent", "critic_agent")

    # Reflection loop router
    graph.add_conditional_edges(
        "critic_agent",
        _route_after_critic,
        {
            "approved": "report_generator",
            "needs_revision": "increment_revision",
        },
    )

    graph.add_edge("increment_revision", "planning_agent")
    graph.add_edge("report_generator", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    checkpointer = SqliteSaver(conn, serde=serde)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


if __name__ == "__main__":
    app_graph = build_graph()
    # Generate unique thread ID for each run to avoid reading stale/corrupted state
    run_id = f"demo-run-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": run_id}}
    
    result = run_pipeline_safely(
        app_graph,
        {"file_path": "sample_data/Sales-Export_2019-2020.csv"},
        config=config,
    )
    print("Revision count:", result.get("revision_count", 0))
    print("Critic feedback:", result.get("critic_feedback"))
    print("\nReport generated at:", result.get("report_path"))