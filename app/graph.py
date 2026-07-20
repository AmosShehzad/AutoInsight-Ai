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
from app.nodes.insight_agent import insight_agent_node   # NEW import

DB_PATH = "checkpoints.sqlite"


def build_graph(db_path: str = DB_PATH, interrupt_after: list[str] | None = None):
    graph = StateGraph(GraphState)

    graph.add_node("file_loader", load_file_node)
    graph.add_node("validation", validate_data_node)
    graph.add_node("cleaning", clean_data_node)
    graph.add_node("profiling", profile_data_node)
    graph.add_node("planning_agent", planning_agent_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insight_agent", insight_agent_node)   # NEW

    graph.set_entry_point("file_loader")

    graph.add_edge("file_loader", "validation")
    graph.add_edge("validation", "cleaning")
    graph.add_edge("cleaning", "profiling")
    graph.add_edge("profiling", "planning_agent")

    # FAN-OUT (from Day 6, unchanged): statistics and visualization run in parallel
    graph.add_edge("planning_agent", "statistics")
    graph.add_edge("planning_agent", "visualization")

    # FAN-IN: both parallel branches now feed into insight_agent instead of END.
    # LangGraph waits for BOTH statistics and visualization to finish
    # before running insight_agent — exactly what we need, since it reads both.
    graph.add_edge("statistics", "insight_agent")
    graph.add_edge("visualization", "insight_agent")
    graph.add_edge("insight_agent", END)   # temporary until Day 8 (Critic Agent)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    checkpointer = SqliteSaver(conn, serde=serde)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


if __name__ == "__main__":
    app_graph = build_graph()
    config = {"configurable": {"thread_id": "demo-run-day7"}}
    result = app_graph.invoke(
        {"file_path": "sample_data/timeseries_sample.csv"},
        config=config,
    )
    print("Plan:", result["plan"])
    print("\nStatistics:", result["statistics"])
    print("\nVisualizations:", result["visualizations"])
    print("\nInsights:")
    for insight in result.get("insights", []):
        print(f"  - {insight}")