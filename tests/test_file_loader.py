import os
import pandas as pd
from app.schemas.state import GraphState


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to auto-detect and convert columns that LOOK like dates
    into real datetime64 dtype.
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            if any(hint in col.lower() for hint in ["date", "time", "day", "month", "year"]):
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() > 0.9:
                    df[col] = converted
    return df


def _parse_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to convert text columns that are ACTUALLY numbers wearing a
    disguise — currency symbols, commas, percent signs — into real
    numeric dtype[cite: 7].
    """
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            # Skip columns already correctly identified as dates
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            cleaned = (
                df[col].astype(str)
                .str.replace(r"[€$£,]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            # Only commit if MOST values converted successfully
            if converted.notna().mean() > 0.9:
                df[col] = converted
    return df


class FileLoadError(Exception):
    """Raised when a file can't be loaded as a dataset."""
    pass


def load_file_node(state: GraphState) -> GraphState:
    """
    LangGraph node: reads file_path from state, loads it into a Pandas 
    DataFrame, and cleans numeric/date columns.
    """
    file_path = state.get("file_path")

    if not file_path:
        raise FileLoadError("No file_path provided in state.")

    if not os.path.exists(file_path):
        raise FileLoadError(f"File not found: {file_path}")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise FileLoadError(f"Unsupported file type: '{extension}'.")
    except pd.errors.EmptyDataError:
        raise FileLoadError("The file is empty.")
    except pd.errors.ParserError:
        raise FileLoadError("The file could not be parsed.")

    # Apply cleaning pipelines
    df = _parse_date_columns(df)
    df = _parse_numeric_columns(df)  # NEW: run numeric cleanup after date parsing[cite: 7]

    if df.empty:
        raise FileLoadError("The loaded dataset has no rows.")

    return {**state, "dataframe": df}