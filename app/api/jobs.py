import os
import re
import shutil
import time
import uuid
from datetime import datetime

from app.graph import PipelineExecutionError, build_graph, run_pipeline_safely
from app.logger import get_logger

logger = get_logger(__name__)

UPLOADS_DIR = "outputs/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Defensive validation pattern matching 12-char hex strings
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")

# The ORDER our graph's nodes run in — used to figure out "which step are we on"
# when reporting status back to the user. Must match app/graph.py's node names.
PIPELINE_STAGES = [
    "file_loader",
    "validation",
    "cleaning",
    "profiling",
    "planning_agent",
    "statistics",
    "visualization",
    "insight_agent",
    "critic_agent",
    "increment_revision",
    "report_generator",
]


def cleanup_old_uploads(max_age_hours: int = 24) -> int:
    """
    Deletes files in UPLOADS_DIR older than `max_age_hours`.
    Returns the count of deleted files. Prevents disk space growth.
    """
    deleted_count = 0
    cutoff_timestamp = time.time() - (max_age_hours * 3600)

    if not os.path.exists(UPLOADS_DIR):
        return 0

    for filename in os.listdir(UPLOADS_DIR):
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.isfile(file_path):
            try:
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_timestamp:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Auto-cleaned old upload file: {filename}")
            except Exception as e:
                logger.error(f"Failed to delete old upload file '{filename}': {e}")

    return deleted_count


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Saves an uploaded file to disk with a unique job_id-based name,
    so two different users uploading 'data.csv' never collide.
    Returns (job_id, saved_file_path).
    """
    job_id = uuid.uuid4().hex[:12]
    extension = os.path.splitext(original_filename)[1]
    saved_path = os.path.join(UPLOADS_DIR, f"{job_id}{extension}")

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Job {job_id}: saved upload '{original_filename}' to {saved_path}")
    return job_id, saved_path


def run_job(job_id: str, file_path: str) -> None:
    """
    Runs the FULL LangGraph pipeline for one job. This function is what
    gets handed to FastAPI's BackgroundTasks — it runs AFTER the endpoint
    has already responded to the user, so uploads feel instant.
    """
    if not JOB_ID_PATTERN.match(job_id):
        logger.error(f"Invalid job_id attempted in run_job: {job_id}")
        return

    # Trigger lightweight cleanup of stale uploads (> 24 hours old)
    try:
        cleanup_old_uploads(max_age_hours=24)
    except Exception as e:
        logger.error(f"Background upload cleanup warning: {e}")

    logger.info(f"Job {job_id}: pipeline started")
    graph = build_graph()
    config = {"configurable": {"thread_id": job_id}}

    try:
        run_pipeline_safely(graph, {"file_path": file_path}, config)
        logger.info(f"Job {job_id}: pipeline completed successfully")
    except PipelineExecutionError as e:
        logger.error(f"Job {job_id}: pipeline failed — {e.stage}: {e.message}")
    except Exception as e:
        logger.error(f"Job {job_id}: unexpected pipeline failure — {e}", exc_info=True)


def get_job_status(job_id: str) -> dict:
    """
    Reads the CURRENT checkpointed state for a job_id, without needing
    the pipeline to be finished.
    """
    if not JOB_ID_PATTERN.match(job_id):
        return {"job_id": job_id, "status": "not_found", "current_stage": None, "progress_percent": 0}

    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": job_id}}
        state_snapshot = graph.get_state(config)
    except Exception as e:
        logger.error(f"Error reading checkpoint state for job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "failed",
            "current_stage": None,
            "progress_percent": 0,
            "error": "An internal error occurred while reading job state.",
        }

    if state_snapshot is None or not state_snapshot.values:
        return {"job_id": job_id, "status": "not_found", "current_stage": None, "progress_percent": 0}

    values = state_snapshot.values

    # If the state recorded an error during pipeline execution, return safe status
    if "error" in values and values["error"]:
        raw_error = values.get("error")
        # Sanitize internal error details/exceptions into clean strings
        user_error = "Pipeline execution encountered an error." if isinstance(raw_error, Exception) else str(raw_error)
        logger.error(f"Job {job_id} reported execution error: {raw_error}")
        return {
            "job_id": job_id,
            "status": "failed",
            "current_stage": None,
            "progress_percent": 0,
            "error": user_error,
        }

    completed_stages = []
    if "dataframe" in values:
        completed_stages.append("file_loader")
    if "validation_report" in values:
        completed_stages.append("validation")
    if "cleaned_dataframe" in values:
        completed_stages.append("cleaning")
    if "profile" in values:
        completed_stages.append("profiling")
    if "plan" in values:
        completed_stages.append("planning_agent")
    if "statistics" in values:
        completed_stages.append("statistics")
    if "visualizations" in values:
        completed_stages.append("visualization")
    if "insights" in values:
        completed_stages.append("insight_agent")
    if "critic_feedback" in values:
        completed_stages.append("critic_agent")
    if "report_path" in values:
        completed_stages.append("report_generator")

    is_done = "report_path" in values
    progress_percent = int((len(completed_stages) / len(PIPELINE_STAGES)) * 100)

    return {
        "job_id": job_id,
        "status": "completed" if is_done else ("processing" if completed_stages else "queued"),
        "current_stage": completed_stages[-1] if completed_stages else None,
        "completed_stages": completed_stages,
        "progress_percent": progress_percent,
        "revision_count": values.get("revision_count", 0),
        "report_path": values.get("report_path") if is_done else None,
    }