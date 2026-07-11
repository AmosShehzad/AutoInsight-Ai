from langgraph.graph import StateGraph, END
from app.schemas.state import GraphState
from app.nodes.file_loader import load_file_node
from app.nodes.validation import validate_data_node
from app.nodes.cleaning import clean_data_node
from app.nodes.profiling import profile_data_node


def build_graph():
    """
    Builds and compiles the LangGraph pipeline.
    Today: purely sequential — File Loader -> Validation -> Cleaning -> Profiling.
    Later days will add parallel branches (Day 6) and conditional loops (Day 8).
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

    return graph.compile()


if __name__ == "__main__":
    # Quick manual run — replace with a real file path to test
    app_graph = build_graph()
    result = app_graph.invoke({"file_path": "sample_data/clean_sample.csv"})
    print("Validation report:", result["validation_report"])
    print("Cleaning report:", result["cleaning_report"])
    print("Profile:", result["profile"])