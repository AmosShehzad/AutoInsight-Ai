from typing import TypedDict, Optional, List
import pandas as pd


class GraphState(TypedDict, total=False):
    """
    The full shared 'memory' for the AutoInsight AI graph.
    Every node in the entire project reads from and writes to this ONE schema.

    Defined fully today (Day 3) even though most fields aren't populated
    until later days (Planning Agent = Day 4, Insight Agent = Day 7, etc.)
    This is intentional: LangGraph works best when the state "shape" is
    known up front, so every node author (future-you) knows exactly what
    keys are safe to read and what keys they're expected to write.
    """

    # --- Input ---
    file_path: str

    # --- Days 1-2: pipeline nodes ---
    dataframe: pd.DataFrame
    validation_report: dict
    cleaned_dataframe: pd.DataFrame
    cleaning_report: dict
    profile: dict

    # --- Day 4-5: Planning Agent + Statistics Node ---
    plan: dict                  # structured output: which analyses/charts to run
    statistics: dict            # Statistics Node's computed results

    # --- Day 6: Visualization Node ---
    visualizations: List[dict]  # list of {chart_type, column(s), file_path}

    # --- Day 7: Insight Generation Agent ---
    insights: List[str]
    insights_detailed: List[dict]   # NEW — full insight objects with source_analysis

    # --- Narrative Agent (business story, column intelligence, opportunities, risks) ---
    narrative: dict
    
    # --- Day 8: Critic Agent + reflection loop ---
    critic_feedback: dict       # {"approved": bool, "reason": str}
    revision_count: int         # how many times Critic sent it back to Planning

    # --- Day 9-10: Report Generation Node ---
    report_path: str

    last_missing_analyses: list