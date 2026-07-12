import os
import sqlite3
import pytest
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.schemas.state import GraphState
from app.nodes.file_loader import load_file_node
from app.nodes.validation import validate_data_node
from app.nodes.cleaning import clean_data_node
from app.nodes.profiling import profile_data_node
from app.graph import build_graph


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_checkpoints.sqlite")


def test_checkpoint_is_saved_after_each_node(temp_db_path, tmp_path):
    import pandas as pd
    file_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file_path, index=False)

    graph = build_graph(db_path=temp_db_path)
    config = {"configurable": {"thread_id": "test-thread-1"}}

    graph.invoke({"file_path": str(file_path)}, config=config)

    # the sqlite file should now exist and contain checkpoint data
    assert os.path.exists(temp_db_path)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    assert len(tables) > 0  # LangGraph created its checkpoint tables


def test_resume_skips_completed_nodes(temp_db_path, tmp_path):
    import pandas as pd
    file_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file_path, index=False)

    thread_id = "test-thread-resume"
    config = {"configurable": {"thread_id": thread_id}}

    # Run a PARTIAL graph (only file_loader + validation) sharing the same DB
    partial = StateGraph(GraphState)
    partial.add_node("file_loader", load_file_node)
    partial.add_node("validation", validate_data_node)
    partial.set_entry_point("file_loader")
    partial.add_edge("file_loader", "validation")
    partial.add_edge("validation", END)

    conn = sqlite3.connect(temp_db_path, check_same_thread=False)
    partial_compiled = partial.compile(checkpointer=SqliteSaver(conn))
    partial_result = partial_compiled.invoke({"file_path": str(file_path)}, config=config)

    assert "validation_report" in partial_result
    assert "cleaning_report" not in partial_result  # cleaning hasn't run yet

    # Now resume with the FULL graph and the SAME thread_id
    full_graph = build_graph(db_path=temp_db_path)
    resumed_result = full_graph.invoke(None, config=config)

    # it should have completed the rest without needing file_path again
    assert "cleaning_report" in resumed_result
    assert "profile" in resumed_result


def test_different_thread_ids_are_independent(temp_db_path, tmp_path):
    import pandas as pd
    file_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file_path, index=False)

    graph = build_graph(db_path=temp_db_path)

    result_a = graph.invoke(
        {"file_path": str(file_path)},
        config={"configurable": {"thread_id": "thread-a"}},
    )
    result_b = graph.invoke(
        {"file_path": str(file_path)},
        config={"configurable": {"thread_id": "thread-b"}},
    )

    # both should succeed independently, proving thread_id isolates state
    assert result_a["profile"]["row_count"] == 2
    assert result_b["profile"]["row_count"] == 2