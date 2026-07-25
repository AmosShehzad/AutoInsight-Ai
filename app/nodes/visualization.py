import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from app.logger import get_logger
from app.schemas.state import GraphState
from app.node_wrapper import node_error_boundary
from app.config import config

logger = get_logger(__name__)

# Single source of truth from config
CHARTS_DIR = Path(config.CHARTS_DIR)

# Modern, boardroom-ready executive color palette
COLORS = {
    "navy": "#0F2540",
    "primary": "#2E5EAA",
    "secondary": "#1B6E7A",
    "gold": "#C9A227",
    "accent": "#E8734A",
    "green": "#2E8B57",
    "grid": "#E5E7EB",
    "text": "#1F2937",
    "subtext": "#6B7280",
}


class VisualizationNodeError(Exception):
    """Raised when the Visualization Node encounters unrecoverable errors."""
    pass


def _apply_chart_style():
    """
    Sets global matplotlib styling once per chart — modern typography,
    soft gridlines, and clean background borders.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["navy"],
        "axes.labelcolor": COLORS["text"],
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,          # High-resolution export for PDF reports
        "savefig.bbox": "tight",
    })


def _format_number(val: float) -> str:
    """Formats large numbers concisely for chart data labels ($1.2M, 450K, etc.)."""
    if abs(val) >= 1_000_000_000:
        return f"{val / 1e9:.2f}B"
    elif abs(val) >= 1_000_000:
        return f"{val / 1e6:.2f}M"
    elif abs(val) >= 1_000:
        return f"{val / 1e3:.1f}K"
    elif val == int(val):
        return f"{int(val):,}"
    else:
        return f"{val:,.2f}"


def _get_primary_metric(df: pd.DataFrame) -> str | None:
    """Finds the primary numeric KPI column in the dataframe."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    keywords = ["cost", "sales", "revenue", "amount", "total", "price", "profit", "net", "gross", "income"]
    for kw in keywords:
        for col in numeric_cols:
            if kw in col.lower():
                return col

    # Exclude ID/code/zip columns
    id_keywords = ["id", "code", "zip", "postal", "year", "number"]
    non_id_cols = [c for c in numeric_cols if not any(id_kw in c.lower() for id_kw in id_keywords)]
    return non_id_cols[0] if non_id_cols else numeric_cols[0]


def _get_primary_categorical(df: pd.DataFrame) -> list[str]:
    """Finds categorical dimensions sorted by usability."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Prioritize columns with reasonable cardinality (between 2 and 30 distinct values)
    usable_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 30]
    return usable_cats if usable_cats else cat_cols


def _get_datetime_column(df: pd.DataFrame) -> str | None:
    """Detects date/datetime columns in the dataframe."""
    dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    if dt_cols:
        return dt_cols[0]

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "month" in col.lower():
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Chart Type Normalization & Dispatch
# ---------------------------------------------------------------------------

def _normalize_chart_type(raw_type: str) -> str:
    """
    Normalizes a chart_type string before alias lookup: lowercases,
    strips whitespace, and collapses spaces/hyphens to underscores.
    """
    return str(raw_type).strip().lower().replace(" ", "_").replace("-", "_")


CHART_TYPE_ALIASES = {
    "bar": "pareto_bar",
    "bar_chart": "pareto_bar",
    "pareto_bar": "pareto_bar",
    "pareto": "pareto_bar",
    "horizontal_bar": "pareto_bar",
    "line": "line",
    "line_chart": "line",
    "time_series": "line",
    "stacked_bar": "stacked_bar",
    "stacked": "stacked_bar",
    "crosstab": "stacked_bar",
    "histogram": "quantile_hist",
    "hist": "quantile_hist",
    "quantile_hist": "quantile_hist",
    "distribution": "quantile_hist",
    "scatter": "scatter",
    "scatter_plot": "scatter",
    "heatmap": "correlation_heatmap",
    "correlation": "correlation_heatmap",
    "correlation_heatmap": "correlation_heatmap",
}


# ---------------------------------------------------------------------------
# Chart Renderers
# ---------------------------------------------------------------------------

def _line_chart(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Time-series trend with 3-period rolling average overlay (resample + rolling)."""
    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        dt_col = _get_datetime_column(df)
        candidate_y = columns[0] if columns else None
        y_col = candidate_y if candidate_y in df.columns and candidate_y != dt_col else _get_primary_metric(df)

        if dt_col and y_col and y_col in df.columns:
            plot_df = df.dropna(subset=[dt_col, y_col]).copy()
            plot_df[dt_col] = pd.to_datetime(plot_df[dt_col])
            plot_df = plot_df.set_index(dt_col).sort_index()

            # Dynamic resampling frequency
            date_span = (plot_df.index.max() - plot_df.index.min()).days if len(plot_df) > 1 else 0
            freq = "ME" if date_span > 90 else "W" if date_span > 14 else "D"
            
            resampled = plot_df[y_col].resample(freq).sum().to_frame()
            resampled["rolling_3p"] = resampled[y_col].rolling(window=3, min_periods=1).mean()

            ax.plot(
                resampled.index, resampled[y_col],
                color=COLORS["primary"], linewidth=2.2, marker="o", markersize=4, label="Period Total"
            )
            ax.plot(
                resampled.index, resampled["rolling_3p"],
                color=COLORS["accent"], linewidth=1.8, linestyle="--", label="3-Period Moving Avg"
            )
            ax.fill_between(resampled.index, resampled[y_col], alpha=0.08, color=COLORS["primary"])
            title_text = f"Time Series Trend: {y_col.replace('_', ' ').title()}"
        else:
            y_col = y_col or df.select_dtypes(include="number").columns[0]
            ax.plot(df.index, df[y_col], color=COLORS["primary"], linewidth=2)
            title_text = f"Trend Line: {y_col.replace('_', ' ').title()}"

        ax.set_title(title_text, pad=12)
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.legend(frameon=True, facecolor="white", edgecolor=COLORS["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


def _pareto_bar_chart(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Horizontal Pareto bar chart for top 10 segment drivers (groupby + nlargest)."""
    cats = _get_primary_categorical(df)
    if columns and columns[0] in df.columns:
        cat_col = columns[0]
    elif cats:
        cat_col = cats[0]
    else:
        raise ValueError("Pareto bar chart requires at least one categorical column.")

    metric_col = columns[1] if len(columns) > 1 and columns[1] in df.columns else _get_primary_metric(df)

    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        if metric_col and metric_col != cat_col:
            grouped = df.groupby(cat_col)[metric_col].sum().nlargest(10).sort_values(ascending=True)
            title = f"Top 10 {cat_col.replace('_', ' ').title()} by Total {metric_col.replace('_', ' ').title()}"
            xlabel = metric_col.replace('_', ' ').title()
        else:
            grouped = df[cat_col].value_counts().nlargest(10).sort_values(ascending=True)
            title = f"Top 10 Categories: {cat_col.replace('_', ' ').title()}"
            xlabel = "Count"

        bars = ax.barh(
            grouped.index.astype(str),
            grouped.values,
            color=COLORS["primary"],
            edgecolor="white",
            height=0.65
        )

        max_val = max(grouped.values) if len(grouped.values) > 0 else 1
        for bar in bars:
            val = bar.get_width()
            ax.text(
                val + (max_val * 0.015),
                bar.get_y() + bar.get_height() / 2,
                _format_number(val),
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["navy"]
            )

        ax.set_title(title, pad=12)
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


def _stacked_bar_chart(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Cross-tabulation stacked bar chart (crosstab + pivot_table)."""
    cats = _get_primary_categorical(df)
    cat1 = columns[0] if columns and columns[0] in df.columns else (cats[0] if cats else None)
    cat2 = columns[1] if len(columns) > 1 and columns[1] in df.columns else (cats[1] if len(cats) > 1 else cat1)

    if not cat1 or not cat2:
        raise ValueError("Stacked bar chart requires 2 valid categorical columns in dataframe.")

    metric = _get_primary_metric(df)

    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        if cat1 != cat2 and metric:
            pivot = pd.pivot_table(
                df, values=metric, index=cat1, columns=cat2, aggfunc="sum", fill_value=0
            )
            top_cats = df[cat1].value_counts().nlargest(7).index
            pivot = pivot.loc[pivot.index.isin(top_cats)]
            title = f"{metric.replace('_', ' ').title()} by {cat1.replace('_', ' ').title()} & {cat2.replace('_', ' ').title()}"
        else:
            pivot = pd.crosstab(df[cat1], df[cat2])
            title = f"Distribution: {cat1.replace('_', ' ').title()} vs {cat2.replace('_', ' ').title()}"

        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="white", width=0.65)

        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace('_', ' ').title() if metric else "Count")
        ax.legend(title=cat2.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


def _quantile_distribution_chart(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Histogram & Boxplot with explicit quantile threshold lines (P50, P90)."""
    col = columns[0] if columns and columns[0] in df.columns else _get_primary_metric(df)
    if not col or col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.empty:
            col = numeric_cols[0]
        else:
            raise ValueError("Quantile distribution chart requires at least 1 numeric column.")

    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        vals = df[col].dropna()
        q50 = float(vals.quantile(0.50))
        q90 = float(vals.quantile(0.90))

        ax.hist(vals, bins=18, color=COLORS["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(q50, color=COLORS["accent"], linestyle="--", linewidth=2, label=f"Median ({_format_number(q50)})")
        ax.axvline(q90, color=COLORS["gold"], linestyle=":", linewidth=2, label=f"P90 ({_format_number(q90)})")

        ax.set_title(f"Distribution & Quantile Bounds: {col.replace('_', ' ').title()}", pad=12)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.legend(frameon=True, facecolor="white", edgecolor=COLORS["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


def _scatter_chart(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Bivariate scatter plot with trend line."""
    if len(columns) < 2:
        raise ValueError(f"Scatter chart requires 2 columns, got {len(columns)}: {columns}")
    x_col, y_col = columns[0], columns[1]
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Scatter chart columns not found in dataframe: {columns}")

    _apply_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        ax.scatter(df[x_col], df[y_col], color=COLORS["secondary"], alpha=0.65, s=50, edgecolors="white", linewidth=0.5)

        # Simple linear trend line
        try:
            valid = df[[x_col, y_col]].dropna()
            z = np.polyfit(valid[x_col], valid[y_col], 1)
            p = np.poly1d(z)
            ax.plot(valid[x_col], p(valid[x_col]), color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Trend Line")
            ax.legend(frameon=True, facecolor="white", edgecolor=COLORS["grid"])
        except Exception:
            pass

        ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}", pad=12)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


def _correlation_heatmap(df: pd.DataFrame, columns: list, save_path: str) -> None:
    """Correlation matrix heatmap with adaptive contrast text values."""
    _apply_chart_style()
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("Correlation heatmap requires at least 2 numeric columns.")

    if numeric_df.shape[1] > 8:
        numeric_df = numeric_df.iloc[:, :8]

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    try:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "#333333",
                    fontsize=8.5, fontweight="medium"
                )

        plt.colorbar(im, label="Correlation", fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels([c.replace("_", " ").title() for c in corr.columns], rotation=35, ha="right")
        ax.set_yticklabels([c.replace("_", " ").title() for c in corr.columns])
        ax.set_title("Numeric Correlation Matrix", pad=12)
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        plt.close(fig)


CHART_RENDERERS = {
    "pareto_bar": _pareto_bar_chart,
    "line": _line_chart,
    "stacked_bar": _stacked_bar_chart,
    "quantile_hist": _quantile_distribution_chart,
    "scatter": _scatter_chart,
    "correlation_heatmap": _correlation_heatmap,
}


# ---------------------------------------------------------------------------
# Auto-Generation Engine (Chart-Heavy Safeguard)
# ---------------------------------------------------------------------------

def _generate_auto_executive_charts(df: pd.DataFrame) -> list[dict]:
    """
    Auto-detects columns and generates 4 to 6 executive-grade charts
    if the plan contains fewer than 4 predefined specifications.
    """
    auto_plan = []
    primary_num = _get_primary_metric(df)
    cats = _get_primary_categorical(df)
    dt_col = _get_datetime_column(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 1. Time-Series Trend Line (if Date + Numeric exist)
    if dt_col and primary_num:
        auto_plan.append({
            "chart_type": "line",
            "columns": [primary_num],
            "reason": f"Time-series trend for {primary_num}"
        })

    # 2. Pareto Top-10 Segment Driver (if Category + Numeric exist)
    if cats and primary_num:
        auto_plan.append({
            "chart_type": "pareto_bar",
            "columns": [cats[0], primary_num],
            "reason": f"Top segment Pareto breakdown of {primary_num} by {cats[0]}"
        })

    # 3. Cross-Tabulation Stacked Bar (if 2 Categories exist)
    if len(cats) >= 2:
        auto_plan.append({
            "chart_type": "stacked_bar",
            "columns": [cats[0], cats[1]],
            "reason": f"Cross-tabulation distribution between {cats[0]} and {cats[1]}"
        })

    # 4. Quantile Distribution Histogram (if Numeric exists)
    if primary_num:
        auto_plan.append({
            "chart_type": "quantile_hist",
            "columns": [primary_num],
            "reason": f"Distribution and quantile threshold spread for {primary_num}"
        })

    # 5. Correlation Heatmap (if >= 2 Numeric cols exist)
    if len(numeric_cols) >= 2:
        auto_plan.append({
            "chart_type": "correlation_heatmap",
            "columns": numeric_cols[:5],
            "reason": "Inter-feature correlation analysis"
        })
    elif len(cats) >= 2 and primary_num:  # Fallback secondary pareto
        auto_plan.append({
            "chart_type": "pareto_bar",
            "columns": [cats[1], primary_num],
            "reason": f"Secondary breakdown of {primary_num} by {cats[1]}"
        })

    return auto_plan


# ---------------------------------------------------------------------------
# LangGraph Node Entrypoint
# ---------------------------------------------------------------------------

@node_error_boundary("visualization")
def visualization_node(state: GraphState) -> dict:
    """
    LangGraph node for generating executive visualizations.
    Renders requested chart specifications and auto-generates fallback executive charts.
    """
    logger.info("Executing Visualization Node...")

    # Safe state extraction & normalization for visualizations
    visualizations = state.get("visualizations")
    if visualizations is None:
        logger.warning("'visualizations' key missing from state — defaulting to empty chart list.")
        visualizations = {"generated": []}
    elif isinstance(visualizations, list):
        logger.warning("'visualizations' was a raw list, not expected dict shape — wrapping it.")
        visualizations = {"generated": visualizations}

    # Flexible DataFrame retrieval
    df: pd.DataFrame = state.get("cleaned_data")
    if df is None or df.empty:
        df = state.get("cleaned_dataframe")
    if df is None or df.empty:
        df = state.get("cleaned_df")
    if df is None or df.empty:
        df = state.get("raw_data")

    if df is None or df.empty:
        logger.error("No DataFrame found in state.")
        raise VisualizationNodeError("Visualization Node requires a valid DataFrame in state.")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plan = state.get("plan", {})
    charts_plan = plan.get("charts", []) if isinstance(plan, dict) else []

    # If plan has fewer than 4 charts, supplement with auto-detected executive charts
    if len(charts_plan) < 4:
        logger.warning(
            f"Plan specified only {len(charts_plan)} chart(s) — supplementing with "
            "auto-generated charts. This is a Visualization Node fallback, not a "
            "Planning Agent decision; tagging auto-generated charts."
        )
        auto_charts = _generate_auto_executive_charts(df)
        
        # Merge without duplicate types
        existing_types = {c.get("chart_type") for c in charts_plan}
        for ac in auto_charts:
            if ac["chart_type"] not in existing_types:
                ac["auto_generated"] = True
                charts_plan.append(ac)

    # Initialize accumulation lists, preserving any existing items in normalized state
    generated = list(visualizations.get("generated", []))
    unsupported_charts = list(visualizations.get("unsupported_charts", []))

    for idx, chart_spec in enumerate(charts_plan):
        raw_chart_type = chart_spec.get("chart_type", "")
        columns = chart_spec.get("columns", [])
        reason = chart_spec.get("reason", "")
        auto_generated = chart_spec.get("auto_generated", False)

        normalized = _normalize_chart_type(raw_chart_type)
        canonical_type = CHART_TYPE_ALIASES.get(normalized)

        if canonical_type is None:
            logger.warning(
                f"Unrecognized chart_type '{raw_chart_type}' (normalized: '{normalized}') — using default fallback 'pareto_bar'."
            )
            canonical_type = "pareto_bar"

        file_name = f"chart_{idx + 1}_{canonical_type}.png"
        file_path = str(CHARTS_DIR / file_name)

        try:
            renderer = CHART_RENDERERS[canonical_type]
            renderer(df, columns, file_path)

            gen_entry = {
                "chart_type": canonical_type,
                "columns": columns,
                "file_path": file_path,
                "reason": reason,
            }
            if auto_generated:
                gen_entry["auto_generated"] = True

            generated.append(gen_entry)
            logger.info(f"Rendered chart [{idx + 1}]: {canonical_type} -> {file_path}")

        except Exception as e:
            logger.error(f"Failed to render chart [{raw_chart_type} / {canonical_type}]: {e}")
            unsupported_charts.append(f"{raw_chart_type}: {str(e)}")

    logger.info(f"Visualization Node completed — {len(generated)} generated, {len(unsupported_charts)} failed.")

    return {
        "visualizations": {
            "generated": generated,
            "unsupported_charts": unsupported_charts,
        }
    }