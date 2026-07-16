import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd

from app.schemas.state import GraphState

# Output directory for generated chart files
CHARTS_DIR = Path("generated_charts")


class VisualizationNodeError(Exception):
    """Raised when the Visualization Node encounters unrecoverable errors."""
    pass


def visualization_node(state: GraphState) -> dict:
    """
    LangGraph node for generating data visualizations.
    Executes chart specifications from state['plan'], saving outputs to disk.
    Gracefully captures missing columns or unsupported chart types without crashing.
    """
    plan = state.get("plan")
    if not plan:
        raise VisualizationNodeError(
            "No 'plan' found in state. Visualization Node requires a plan."
        )

    df: pd.DataFrame = state.get("cleaned_dataframe")
    if df is None or df.empty:
        raise VisualizationNodeError(
            "No 'cleaned_dataframe' found in state or DataFrame is empty."
        )

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    charts_plan = plan.get("charts", [])
    generated = []
    unsupported_charts = []

    for idx, chart_spec in enumerate(charts_plan):
        chart_type = chart_spec.get("chart_type")
        columns = chart_spec.get("columns", [])
        reason = chart_spec.get("reason", "")

        # Guardrail: Check if referenced columns actually exist in DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            unsupported_charts.append(
                f"{chart_type}: Missing column(s) {missing_cols}"
            )
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        chart_created = False

        try:
            if chart_type == "bar":
                col = columns[0]
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                chart_created = True

            elif chart_type == "line":
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    ax.plot(df[x_col], df[y_col])
                    ax.set_title(f"{y_col} over {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    col = columns[0]
                    ax.plot(df[col])
                    ax.set_title(f"Line Chart of {col}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel(col)
                chart_created = True

            elif chart_type == "histogram":
                col = columns[0]
                df[col].plot(kind="hist", ax=ax, bins=15)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                chart_created = True

            else:
                unsupported_charts.append(chart_type)

            if chart_created:
                file_name = f"chart_{idx + 1}_{chart_type}.png"
                file_path = str(CHARTS_DIR / file_name)
                plt.tight_layout()
                fig.savefig(file_path)

                generated.append({
                    "chart_type": chart_type,
                    "columns": columns,
                    "file_path": file_path,
                    "reason": reason,
                })

        except Exception as e:
            unsupported_charts.append(f"{chart_type}: {str(e)}")
        finally:
            plt.close(fig)

    # Returns ONLY the key updated by this node
    return {
        "visualizations": {
            "generated": generated,
            "unsupported_charts": unsupported_charts,
        }
    }