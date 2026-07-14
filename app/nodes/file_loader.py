import os
import pandas as pd
from app.schemas.state import GraphState


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to auto-detect and convert columns that LOOK like dates
    (based on column name hints and successful parsing) into real
    datetime64 dtype, so downstream nodes (Profiling, Planning Agent,
    Statistics trend_analysis) can correctly recognize them as dates.
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Only attempt parsing if the column name hints at a date,
            # to avoid accidentally converting unrelated text columns
            if any(hint in col.lower() for hint in ["date", "time", "day", "month", "year"]):
                converted = pd.to_datetime(df[col], errors="coerce")

                # Only commit the conversion if MOST values parsed successfully
                if converted.notna().mean() > 0.9:
                    df[col] = converted

    return df


class FileLoadError(Exception):
    """Raised when a file can't be loaded as a dataset."""
    pass


def load_file_node(state: GraphState) -> GraphState:
    """
    LangGraph node: reads file_path from state, detects CSV or Excel,
    loads it into a Pandas DataFrame, and writes it back into state.
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
            raise FileLoadError(
                f"Unsupported file type: '{extension}'. Only .csv, .xlsx, .xls are supported."
            )
    except pd.errors.EmptyDataError:
        raise FileLoadError("The file is empty.")
    except pd.errors.ParserError:
        raise FileLoadError("The file could not be parsed. It may be corrupted or malformed.")

    # Convert date-like text columns into datetime dtype
    df = _parse_date_columns(df)

    if df.empty:
        raise FileLoadError("The loaded dataset has no rows.")

    return {**state, "dataframe": df}