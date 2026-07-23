import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import pandas as pd

from app.api.jobs import save_uploaded_file, run_job, get_job_status
from app.graph import build_graph
from app.logger import get_logger

logger = get_logger(__name__)

# This single line creates the whole FastAPI application object.
# Everything below (@app.get, @app.post) attaches endpoints to it.
app = FastAPI(
    title="AutoInsight AI API",
    description="Upload a dataset, get back a cleaned dataset, EDA, and an executive PDF report.",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@app.get("/")
def read_root():
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
        # HTTPException is FastAPI's way of returning a proper error
        # response (with a status code) instead of a Python crash.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Only .csv, .xlsx, .xls are allowed.",
        )

    file_bytes = await file.read()   # 'await' because reading an uploaded file is an async operation
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id, saved_path = save_uploaded_file(file_bytes, file.filename)

    # add_task schedules run_job to execute AFTER this function returns
    # its response — this is what makes the upload feel instant.
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
    from LangGraph's checkpoint, so this works WHILE the job is still
    running in the background.
    """
    status = get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")
    return status


@app.get("/report/{job_id}")
def download_report(job_id: str):
    """
    Serves the finished PDF report for download. Returns a clear 404 if
    the job doesn't exist, and a clear 409 (conflict) if the job exists
    but hasn't finished yet — never a confusing missing-file error.
    """
    status = get_job_status(job_id)

    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")
    if status["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not finished yet (current stage: {status['current_stage']}).",
        )

    report_path = status["report_path"]
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=500, detail="Report file is missing on the server.")

    # FileResponse streams the actual file bytes back as the HTTP response,
    # with the right headers so a browser/client treats it as a downloadable file.
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"autoinsight_report_{job_id}.pdf",
    )


@app.get("/cleaned-data/{job_id}")
def download_cleaned_data(job_id: str):
    """
    Serves the cleaned dataset (from state['cleaned_dataframe']) as a
    downloadable CSV file. Reads the dataframe straight from the
    checkpoint, converts it to CSV on the fly — no separate file needed
    on disk for this.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": job_id}}
    state_snapshot = graph.get_state(config)

    if state_snapshot is None or not state_snapshot.values:
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")

    cleaned_df = state_snapshot.values.get("cleaned_dataframe")
    if cleaned_df is None:
        raise HTTPException(
            status_code=409,
            detail="Cleaned data isn't ready yet — the pipeline hasn't reached the Cleaning step.",
        )

    output_path = f"outputs/uploads/{job_id}_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)

    return FileResponse(
        path=output_path,
        media_type="text/csv",
        filename=f"autoinsight_cleaned_{job_id}.csv",
    )