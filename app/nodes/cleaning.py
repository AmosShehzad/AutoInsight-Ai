import pandas as pd
from app.schemas.state import GraphState


def clean_data_node(state: GraphState) -> GraphState:
    """
    LangGraph node: takes the raw dataframe, handles missing values,
    removes duplicate rows, and corrects obvious dtype issues.
    Writes cleaned_dataframe + cleaning_report back into state.
    """
    df: pd.DataFrame = state["dataframe"]
    original_rows = len(df)

    cleaned = df.copy()

    # 1. Remove exact duplicate rows
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # 2. Handle missing values column by column
    missing_handling = {}
    for col in cleaned.columns:
        missing_count = int(cleaned[col].isnull().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[col]):
            fill_value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_value)
            missing_handling[col] = {
                "strategy": "median_fill",
                "fill_value": float(fill_value),
                "count_filled": missing_count,
            }
        else:
            fill_value = "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
            missing_handling[col] = {
                "strategy": "constant_fill",
                "fill_value": fill_value,
                "count_filled": missing_count,
            }

    # 3. Try to correct obvious dtype issues
    # e.g. a numeric column that got loaded as text because of stray characters
    dtype_corrections = {}
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            # if converting to numeric doesn't blow up more than 5% of values to NaN,
            # it was probably meant to be numeric all along
            non_convertible_ratio = converted.isnull().mean()
            if non_convertible_ratio < 0.05 and not converted.isnull().all():
                cleaned[col] = converted.fillna(converted.median())
                dtype_corrections[col] = "object_to_numeric"

    cleaning_report = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "rows_after_cleaning": len(cleaned),
        "missing_value_handling": missing_handling,
        "dtype_corrections": dtype_corrections,
    }

    return {**state, "cleaned_dataframe": cleaned, "cleaning_report": cleaning_report}