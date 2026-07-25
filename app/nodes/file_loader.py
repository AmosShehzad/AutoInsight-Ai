import os
import pandas as pd
from app.schemas.state import GraphState
from app.logger import get_logger

logger = get_logger(__name__)


class FileLoadError(Exception):
    """Raised when a file can't be loaded as a dataset."""
    pass


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects object/string columns that match date keywords and converts
    them to datetime if at least 80% of non-null values can be parsed.
    """
    df = df.copy()
    date_keywords = [
        "date", "time", "day", "month", "year", 
        "dt", "timestamp", "period", "quarter"
    ]
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            col_lower = str(col).lower()
            if any(hint in col_lower for hint in date_keywords):
                try:
                    converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
                except TypeError:
                    converted = pd.to_datetime(df[col], errors="coerce")
                
                non_null_count = df[col].dropna().count()
                if non_null_count > 0:
                    valid_ratio = converted.notna().sum() / non_null_count
                    if valid_ratio >= 0.8:
                        df[col] = converted
                        logger.info(f"Converted column '{col}' to datetime (valid ratio: {valid_ratio:.2%}).")
                        
    return df

from app.node_wrapper import node_error_boundary

@node_error_boundary("file_loader")
def load_file_node(state: GraphState) -> GraphState:
    """
    LangGraph node: reads file_path from state, detects CSV or Excel,
    loads it into a Pandas DataFrame with multi-encoding fallback support, 
    strips whitespace from column names, parses date columns, and updates state.
    """
    file_path = state.get("file_path")

    if not file_path:
        logger.error("No file_path provided in state.")
        raise FileLoadError("No file_path provided in state.")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileLoadError(f"File not found: {file_path}")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    logger.info(f"Attempting to load file: {file_path} (Extension: {extension})")

    df = None

    try:
        if extension == ".csv":
            # Sequential fallback mechanism for tricky text encodings
            encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    logger.info(f"Successfully read CSV using '{enc}' encoding.")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                raise FileLoadError(
                    f"Failed to decode CSV file using encodings: {', '.join(encodings_to_try)}."
                )

        elif extension in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
            logger.info("Successfully read Excel file.")
        else:
            raise FileLoadError(
                f"Unsupported file type: '{extension}'. Only .csv, .xlsx, and .xls are supported."
            )

    except pd.errors.EmptyDataError:
        logger.error(f"Empty data error for file: {file_path}")
        raise FileLoadError("The file is empty.")
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while reading file: {file_path}. Details: {e}")
        raise FileLoadError(f"The file could not be parsed: {e}")
    except Exception as e:
        if isinstance(e, FileLoadError):
            raise e
        logger.error(f"Unexpected error loading file {file_path}: {e}")
        raise FileLoadError(f"Failed to load file due to error: {e}")

    if df is None or df.empty:
        logger.error("Loaded DataFrame is empty.")
        raise FileLoadError("The loaded dataset has no rows or columns.")

    # Strip hidden leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()

    # Convert date-like text columns into datetime dtype
    df = _parse_date_columns(df)

    logger.info(f"File successfully loaded into DataFrame. Dimensions: {df.shape[0]} rows x {df.shape[1]} columns.")

    return {**state, "dataframe": df}