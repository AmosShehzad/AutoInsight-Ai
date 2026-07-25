import os
import re
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.jobs import save_uploaded_file, run_job, get_job_status
from app.graph import build_graph
from app.logger import get_logger

logger = get_logger(__name__)

# FastAPI application initialization
app = FastAPI(
    title="AutoInsight AI API",
    description="Upload a dataset, get back a cleaned dataset, EDA, and an executive PDF report.",
    version="1.0.0",
)

# Serves everything in app/static/ at the URL path /static/...
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Enable Cross-Origin Resource Sharing (CORS) for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to specific frontend origin in production
    allow_credentials=False,  # Spec-compliant with wildcard origins
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_SIZE_MB = 50

# Strict 12-char hex pattern matching uuid.uuid4().hex[:12] from save_uploaded_file
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def _validate_job_id(job_id: str) -> None:
    """
    Validates job_id shape to prevent Path Traversal attacks (e.g. '../../etc/passwd').
    Raises HTTP 400 Bad Request if the job_id doesn't strictly match expected format.
    """
    if not job_id or not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job_id format.",
        )


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the frontend's index.html when someone visits the root URL."""
    return FileResponse("app/static/index.html")


@app.get("/health")
def health_check():
    """Simple health-check endpoint — confirms the API is running at all."""
    return {"message": "AutoInsight AI API is running.", "docs": "/docs"}


@app.post("/upload")
async def upload_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a CSV or Excel file, saves it, and starts the pipeline
    running in the BACKGROUND. Responds immediately with a job_id —
    the caller then polls /status/{job_id} to check progress.
    """
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Only .csv, .xlsx, .xls are allowed.",
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file '{file.filename}': {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded file content.",
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        job_id, saved_path = save_uploaded_file(file_bytes, file.filename)
    except Exception as e:
        logger.error(f"Failed to save upload '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while saving the uploaded file.",
        )

    # Schedules run_job to execute in background after response returns
    background_tasks.add_task(run_job, job_id, saved_path)

    logger.info(f"Job {job_id}: upload accepted, pipeline queued")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "File received. Poll /status/{job_id} for progress.",
    }


@app.get("/status/{job_id}")
def check_status(job_id: str):
    """
    Reports current progress for a job — which pipeline stage it's on,
    how many stages are done, and whether it's finished. Reads directly
    from LangGraph's checkpoint.
    """
    _validate_job_id(job_id)

    try:
        status = get_job_status(job_id)
    except Exception as e:
        logger.error(f"Status check failed for job {job_id}: {e}")
        # Return a "still processing, try again" response instead of a hard 500 —
        # a transient DB hiccup shouldn't look like a permanent failure to the user.
        return {
            "job_id": job_id,
            "status": "processing",
            "current_stage": None,
            "progress_percent": 0,
            "note": "Status temporarily unavailable, retrying...",
        }

    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")
    return status


@app.get("/report/{job_id}")
def download_report(job_id: str):
    """
    Serves the finished PDF report for download. Returns 404 if missing,
    or 409 Conflict if the job is still running.
    """
    _validate_job_id(job_id)

    try:
        status = get_job_status(job_id)
    except Exception as e:
        logger.error(f"Error checking status during report request for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving report information.",
        )

    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")
    if status["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not finished yet (current stage: {status['current_stage']}).",
        )

    report_path = status["report_path"]
    if not report_path or not os.path.exists(report_path):
        logger.error(f"Report file missing on server for job {job_id} at path '{report_path}'")
        raise HTTPException(status_code=500, detail="Report file is missing on the server.")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"autoinsight_report_{job_id}.pdf",
    )


@app.get("/cleaned-data/{job_id}")
def download_cleaned_data(job_id: str):
    """
    Serves the cleaned dataset as a downloadable CSV file by reading
    state['cleaned_dataframe'] directly from the state snapshot.
    """
    _validate_job_id(job_id)

    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": job_id}}
        state_snapshot = graph.get_state(config)
    except Exception as e:
        logger.error(f"Error accessing graph checkpoint for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving cleaned dataset.",
        )

    if state_snapshot is None or not state_snapshot.values:
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")

    cleaned_df = state_snapshot.values.get("cleaned_dataframe")
    if cleaned_df is None:
        raise HTTPException(
            status_code=409,
            detail="Cleaned data isn't ready yet — the pipeline hasn't reached the Cleaning step.",
        )

    try:
        output_path = f"outputs/uploads/{job_id}_cleaned.csv"
        cleaned_df.to_csv(output_path, index=False)

        return FileResponse(
            path=output_path,
            media_type="text/csv",
            filename=f"autoinsight_cleaned_{job_id}.csv",
        )
    except Exception as e:
        logger.error(f"Error exporting cleaned dataset to CSV for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the CSV file.",
        )