import pandas as pd
from app.schemas.state import GraphState
from app.logger import get_logger

logger = get_logger(__name__)


def clean_data_node(state: GraphState) -> GraphState:
    """
    LangGraph node: takes the raw dataframe, strips formatting symbols (currency, commas, %),
    handles missing values, removes duplicate rows, corrects numeric dtypes, and trims column names.
    
    Writes cleaned_dataframe + cleaning_report back into state.
    """
    df: pd.DataFrame = state.get("dataframe")

    if df is None or df.empty:
        logger.warning("clean_data_node received an empty or missing DataFrame.")
        return {
            **state,
            "cleaned_dataframe": pd.DataFrame(),
            "cleaning_report": {
                "original_rows": 0,
                "duplicates_removed": 0,
                "rows_after_cleaning": 0,
                "missing_value_handling": {},
                "dtype_corrections": {},
            },
        }

    original_rows = len(df)
    cleaned = df.copy()

    # Strip hidden leading/trailing spaces from column headers
    cleaned.columns = cleaned.columns.str.strip()

    logger.info(f"Starting data cleaning on dataset with {original_rows} rows and {len(cleaned.columns)} columns.")

    # 1. Remove exact duplicate rows
    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed > 0:
        cleaned = cleaned.drop_duplicates()
        logger.info(f"Removed {duplicates_removed} duplicate rows.")

    # 2. Pre-clean text-formatted currency, percentage, and formatted numeric columns
    dtype_corrections = {}
    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in object_cols:
        non_null_series = cleaned[col].dropna().astype(str).str.strip()
        if non_null_series.empty:
            continue

        # Inspect non-null string samples for currency symbols, commas, or percent signs
        sample = non_null_series.head(50)
        pattern = r"^[\$\€\£\¥]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?%?$|^[\$\€\£\¥]?\s*-?\d+(\.\d+)?%?$"
        matches = sample.str.match(pattern)

        if matches.mean() > 0.5:
            # Strip currency symbols, commas, percent signs, and trailing whitespace
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.replace(r"[\$\€\£\¥\,\%]", "", regex=True)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned[col], errors="coerce")

            # Adopt numeric dtype if majority of values parse successfully
            if converted.notna().mean() > 0.5:
                cleaned[col] = converted
                dtype_corrections[col] = "formatted_string_to_numeric"
                logger.info(f"Successfully cleaned formatted currency/numeric text in column '{col}'.")

    # 3. Handle missing values column by column
    missing_handling = {}
    for col in cleaned.columns:
        missing_count = int(cleaned[col].isnull().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[col]):
            non_null_vals = cleaned[col].dropna()
            fill_val = float(non_null_vals.median()) if not non_null_vals.empty else 0.0
            cleaned[col] = cleaned[col].fillna(fill_val)
            missing_handling[col] = {
                "strategy": "median_fill",
                "fill_value": fill_val,
                "count_filled": missing_count,
            }
            logger.info(f"Filled {missing_count} missing values in numeric column '{col}' with median ({fill_val}).")
        
        elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            # Retain NaT for datetime columns to prevent corrupting time metrics
            missing_handling[col] = {
                "strategy": "retained_nat",
                "fill_value": "NaT",
                "count_filled": missing_count,
            }
            logger.info(f"Retained {missing_count} missing values as NaT in datetime column '{col}'.")
            
        else:
            fill_val = "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)
            missing_handling[col] = {
                "strategy": "constant_fill",
                "fill_value": fill_val,
                "count_filled": missing_count,
            }
            logger.info(f"Filled {missing_count} missing values in string column '{col}' with '{fill_val}'.")

    # 4. Fallback numeric coercion check for remaining string columns
    remaining_object_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in remaining_object_cols:
        if col in dtype_corrections:
            continue
        converted = pd.to_numeric(cleaned[col], errors="coerce")
        non_convertible_ratio = converted.isnull().mean()
        
        # If less than 5% of non-convertible values exist, coerce to numeric
        if non_convertible_ratio < 0.05 and not converted.isnull().all():
            non_null_vals = converted.dropna()
            fill_val = float(non_null_vals.median()) if not non_null_vals.empty else 0.0
            cleaned[col] = converted.fillna(fill_val)
            dtype_corrections[col] = "object_to_numeric"
            logger.info(f"Converted column '{col}' from text to numeric.")

    cleaning_report = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "rows_after_cleaning": len(cleaned),
        "missing_value_handling": missing_handling,
        "dtype_corrections": dtype_corrections,
    }

    logger.info(
        f"Data cleaning completed successfully. Rows: {original_rows} -> {len(cleaned)}. "
        f"Columns converted: {len(dtype_corrections)}."
    )

    return {**state, "cleaned_dataframe": cleaned, "cleaning_report": cleaning_report}