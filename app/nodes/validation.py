"""
Validation Node for AutoInsight AI
Inspects raw or incoming DataFrames for structural integrity, completeness,
duplicates, and data hygiene issue flags.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from app.logger import get_logger
from app.schemas.state import GraphState

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails or input is invalid."""
    pass


def _calculate_health_score(
    df: pd.DataFrame,
    total_cells: int,
    missing_cells: int,
    duplicate_rows: int,
    empty_cols: List[str],
    constant_cols: List[str]
) -> Tuple[float, str, List[str]]:
    """
    Computes a multi-factor health score (0.0 to 100.0), letter grade,
    and a list of actionable quality warnings.
    """
    total_rows, total_cols = df.shape
    if total_rows == 0 or total_cols == 0:
        return 0.0, "F", ["Dataset is completely empty."]

    missing_pct = (missing_cells / total_cells) * 100.0 if total_cells else 0.0
    duplicate_pct = (duplicate_rows / total_rows) * 100.0 if total_rows else 0.0
    empty_cols_pct = (len(empty_cols) / total_cols) * 100.0 if total_cols else 0.0
    constant_cols_pct = (len(constant_cols) / total_cols) * 100.0 if total_cols else 0.0

    # Weighted Penalties (Max combined penalty capped at 100)
    missing_penalty = min(35.0, missing_pct * 0.7)
    duplicate_penalty = min(25.0, duplicate_pct * 0.5)
    empty_col_penalty = min(25.0, empty_cols_pct * 1.0)
    constant_penalty = min(15.0, constant_cols_pct * 0.5)

    total_penalty = missing_penalty + duplicate_penalty + empty_col_penalty + constant_penalty
    health_score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 2)

    # Letter Grade
    if health_score >= 90.0:
        grade = "A"
    elif health_score >= 75.0:
        grade = "B"
    elif health_score >= 60.0:
        grade = "C"
    elif health_score >= 40.0:
        grade = "D"
    else:
        grade = "F"

    # Actionable Warnings
    warnings = []
    if missing_pct > 10.0:
        warnings.append(f"High cell missingness detected ({missing_pct:.1f}% missing).")
    if duplicate_pct > 5.0:
        warnings.append(f"Significant duplicate rows found ({duplicate_pct:.1f}% duplicate).")
    if empty_cols:
        warnings.append(f"Contains {len(empty_cols)} completely empty column(s): {empty_cols}.")
    if constant_cols:
        warnings.append(f"Contains {len(constant_cols)} zero-variance column(s): {constant_cols}.")
    if total_rows < 10:
        warnings.append(f"Extremely small dataset size ({total_rows} rows). Statistical analyses may be unviable.")

    return health_score, grade, warnings


def _format_memory_usage(bytes_val: int) -> str:
    """Formats byte count into human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} Bytes"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def validate_data_node(state: GraphState) -> GraphState:
    """
    LangGraph Node: Inspects state DataFrame for missing values, duplicates,
    data types, zero-variance columns, and calculates a data health score.
    """
    df: pd.DataFrame = state.get("dataframe")

    if df is None:
        logger.error("Validation Node failed: 'dataframe' missing from state.")
        raise DataValidationError("State does not contain 'dataframe'.")

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Validation Node failed: expected pandas DataFrame, got {type(df)}.")
        raise DataValidationError(f"Invalid state data type: {type(df)}.")

    total_rows, total_cols = df.shape
    logger.info(f"Validation Node started — {total_rows} rows, {total_cols} columns")

    if df.empty:
        logger.warning("Dataset is empty. Skipping detailed validation checks.")
        validation_report = {
            "row_count": 0,
            "column_count": 0,
            "missing_cells": 0,
            "missing_percent": 0.0,
            "duplicate_rows": 0,
            "duplicate_percent": 0.0,
            "empty_columns": [],
            "constant_columns": [],
            "dtype_summary": {},
            "memory_usage": "0 KB",
            "health_score": 0.0,
            "health_grade": "F",
            "warnings": ["Dataset is completely empty."],
        }
        return {**state, "validation_report": validation_report}

    total_cells = total_rows * total_cols
    missing_cells = int(df.isnull().sum().sum())
    missing_percent = round((missing_cells / total_cells) * 100.0, 2) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = round((duplicate_rows / total_rows) * 100.0, 2) if total_rows else 0.0

    dtype_summary = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    
    # Identify empty and constant (zero variance) columns
    empty_cols = [str(col) for col in df.columns if df[col].isnull().all()]
    constant_cols = [
        str(col) for col in df.columns 
        if df[col].nunique(dropna=True) <= 1 and str(col) not in empty_cols
    ]

    # Memory usage
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_str = _format_memory_usage(memory_bytes)

    # Health Score Calculation
    health_score, health_grade, warnings = _calculate_health_score(
        df, total_cells, missing_cells, duplicate_rows, empty_cols, constant_cols
    )

    validation_report = {
        "row_count": total_rows,
        "column_count": total_cols,
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "empty_columns": empty_cols,
        "constant_columns": constant_cols,
        "dtype_summary": dtype_summary,
        "memory_usage": memory_str,
        "health_score": health_score,
        "health_grade": health_grade,
        "warnings": warnings,
    }

    logger.info(f"Validation complete — Health Score: {health_score}/100 (Grade {health_grade})")
    if warnings:
        logger.warning(f"Validation Warnings: {warnings}")

    return {**state, "validation_report": validation_report}