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

DB_PATH = "checkpoints.sqlite"


def build_graph(db_path: str = DB_PATH, interrupt_after: list[str] | None = None):
    """
    Builds and compiles the LangGraph pipeline WITH checkpointing.

    interrupt_after: optional list of node names to pause after —
    used only for the crash-simulation demo, not real usage.
    """
    graph = StateGraph(GraphState)

    graph.add_node("file_loader", load_file_node)
    graph.add_node("validation", validate_data_node)
    graph.add_node("cleaning", clean_data_node)
    graph.add_node("profiling", profile_data_node)

    graph.set_entry_point("file_loader")

    graph.add_edge("file_loader", "validation")
    graph.add_edge("validation", "cleaning")
    graph.add_edge("cleaning", "profiling")
    graph.add_edge("profiling", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    checkpointer = SqliteSaver(conn, serde=serde)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)

if __name__ == "__main__":
    app_graph = build_graph()
    config = {"configurable": {"thread_id": "demo-run-1"}}
    result = app_graph.invoke(
        {"file_path": "sample_data/clean_sample.csv"},
        config=config,
    )
    print("Profile:", result["profile"])