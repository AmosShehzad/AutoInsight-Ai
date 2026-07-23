import os
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

requires_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live pipeline tests"
)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_upload_rejects_unsupported_file_type(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")

    with open(bad_file, "rb") as f:
        response = client.post("/upload", files={"file": ("notes.txt", f, "text/plain")})

    assert response.status_code == 400


def test_upload_rejects_empty_file():
    response = client.post("/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_status_returns_404_for_unknown_job():
    response = client.get("/status/does-not-exist-123")
    assert response.status_code == 404


def test_report_returns_404_for_unknown_job():
    response = client.get("/report/does-not-exist-123")
    assert response.status_code == 404


@requires_api_key
def test_full_upload_and_poll_flow(tmp_path):
    """
    THE key end-to-end test: upload a real file, poll status until
    done (with a timeout), then confirm the report can be downloaded.
    """
    csv_path = "sample_data/clean_sample.csv"
    with open(csv_path, "rb") as f:
        upload_response = client.post("/upload", files={"file": ("clean_sample.csv", f, "text/csv")})

    assert upload_response.status_code == 200
    job_id = upload_response.json()["job_id"]

    # Poll status until completed, or give up after ~60 seconds
    for _ in range(30):
        status_response = client.get(f"/status/{job_id}")
        assert status_response.status_code == 200
        if status_response.json()["status"] == "completed":
            break
        time.sleep(2)
    else:
        pytest.fail("Job did not complete within the timeout window")

    report_response = client.get(f"/report/{job_id}")
    assert report_response.status_code == 200
    assert report_response.headers["content-type"] == "application/pdf"

    cleaned_response = client.get(f"/cleaned-data/{job_id}")
    assert cleaned_response.status_code == 200