"""
Manual test script: simulates a crash after the Validation node,
then shows that a second run with the SAME thread_id resumes
instead of starting over.

Run this file twice in a row (see instructions below).
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.graph import build_graph

THREAD_ID = "crash-test-1"


def run_first_half():
    """
    Simulates the process being killed after validation.
    Uses the SAME real 4-node graph, but tells LangGraph to
    pause (interrupt) right after the 'validation' node runs.
    """
    graph = build_graph(interrupt_after=["validation"])
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = graph.invoke(
        {"file_path": "sample_data/clean_sample.csv"},
        config=config,
    )
    print("First half done. Validation report saved:")
    print(result["validation_report"])
    print("\n--- Now 'crashing' (process paused here) ---")


def run_full_resume():
    """
    Simulates restarting the process later and resuming.
    Same real graph, no interrupt this time, so it runs to completion.
    Passing None resumes from the last checkpoint instead of restarting.
    """
    full_graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = full_graph.invoke(None, config=config)

    print("Resumed and completed. Full state now includes:")
    print("Validation report:", result["validation_report"])
    print("Cleaning report:", result["cleaning_report"])
    print("Profile:", result["profile"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "first"
    if mode == "first":
        run_first_half()
    elif mode == "resume":
        run_full_resume()