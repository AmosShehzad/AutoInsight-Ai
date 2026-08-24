"""
Central configuration for AutoInsight AI. All tunable values live here,
loaded from environment variables with sensible defaults. This is the
ONE place to change model names, limits, or paths — never hardcode
these values directly in node files.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- LLM settings ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("AUTOINSIGHT_LLM_MODEL", "openai/gpt-oss-20b")

    # Per-agent temperature — different agents want different creativity levels,
    # but the MODEL should be consistent unless explicitly overridden.
    PLANNING_TEMPERATURE: float = float(os.getenv("AUTOINSIGHT_PLANNING_TEMP", "0.2"))
    INSIGHT_TEMPERATURE: float = float(os.getenv("AUTOINSIGHT_INSIGHT_TEMP", "0.3"))
    CRITIC_TEMPERATURE: float = float(os.getenv("AUTOINSIGHT_CRITIC_TEMP", "0.1"))

    # --- LangSmith observability ---
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "autoinsight-ai")

    # --- Reflection loop ---
    MAX_REVISIONS: int = int(os.getenv("AUTOINSIGHT_MAX_REVISIONS", "2"))
    MAX_LLM_RETRIES: int = int(os.getenv("AUTOINSIGHT_MAX_LLM_RETRIES", "2"))

    # --- Storage paths ---
    DB_PATH: str = os.getenv("AUTOINSIGHT_DB_PATH", "checkpoints.sqlite")
    CHARTS_DIR: str = os.getenv("AUTOINSIGHT_CHARTS_DIR", "outputs/charts")
    REPORTS_DIR: str = os.getenv("AUTOINSIGHT_REPORTS_DIR", "outputs/reports")
    UPLOADS_DIR: str = os.getenv("AUTOINSIGHT_UPLOADS_DIR", "outputs/uploads")
    LOGS_DIR: str = os.getenv("AUTOINSIGHT_LOGS_DIR", "outputs/logs")


config = Config()