import os
import uuid
import shutil
from datetime import datetime

from app.graph import build_graph, run_pipeline_safely, PipelineExecutionError
from app.logger import get_logger

logger = get_logger(__name__)

UPLOADS_DIR = "outputs/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# The ORDER our graph's nodes run in — used to figure out "which step are we on"
# when reporting status back to the user. Must match app/graph.py's node names.
PIPELINE_STAGES = [
    "file_loader", "validation", "cleaning", "profiling", "planning_agent",
    "statistics", "visualization", "insight_agent", "critic_agent",
    "increment_revision", "report_generator",
]


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
    logger.info(f"Job {job_id}: pipeline started")
    graph = build_graph()
    config = {"configurable": {"thread_id": job_id}}

    try:
        run_pipeline_safely(graph, {"file_path": file_path}, config)
        logger.info(f"Job {job_id}: pipeline completed successfully")
    except PipelineExecutionError as e:
        # The error is already logged inside run_pipeline_safely / the nodes.
        # We don't re-raise here because this runs in a background thread —
        # an uncrashed background task means /status can still report failure cleanly.
        logger.error(f"Job {job_id}: pipeline failed — {e.stage}: {e.message}")


def get_job_status(job_id: str) -> dict:
    """
    Reads the CURRENT checkpointed state for a job_id, without needing
    the pipeline to be finished. This is what makes '/status/{job_id}'
    possible — we ask LangGraph's checkpointer "what does this thread's
    state look like right now?"
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": job_id}}

    # get_state() reads the LATEST checkpoint for this thread_id — this
    # is a read-only operation, it does NOT re-run anything.
    state_snapshot = graph.get_state(config)

    if state_snapshot is None or not state_snapshot.values:
        return {"job_id": job_id, "status": "not_found", "current_stage": None, "progress_percent": 0}

    values = state_snapshot.values

    # Figure out which stage we're on by checking which state keys exist —
    # e.g. if 'report_path' exists, we're done; if only 'profile' exists,
    # we're right after Profiling and about to run Planning Agent.
    completed_stages = []
    if "dataframe" in values: completed_stages.append("file_loader")
    if "validation_report" in values: completed_stages.append("validation")
    if "cleaned_dataframe" in values: completed_stages.append("cleaning")
    if "profile" in values: completed_stages.append("profiling")
    if "plan" in values: completed_stages.append("planning_agent")
    if "statistics" in values: completed_stages.append("statistics")
    if "visualizations" in values: completed_stages.append("visualization")
    if "insights" in values: completed_stages.append("insight_agent")
    if "critic_feedback" in values: completed_stages.append("critic_agent")
    if "report_path" in values: completed_stages.append("report_generator")

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