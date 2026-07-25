// ---------------------------------------------------------------------------
// Config — the base URL of your FastAPI backend. Same origin here since
// FastAPI is serving this HTML file itself; change this if you ever host
// the frontend separately from the API.
// ---------------------------------------------------------------------------
const API_BASE = "";

// Must match app/api/jobs.py's PIPELINE_STAGES list, in the same order —
// this is how we draw the live progress checklist.
const PIPELINE_STAGES = [
    { key: "file_loader", label: "Loading file" },
    { key: "validation", label: "Validating data" },
    { key: "cleaning", label: "Cleaning data" },
    { key: "profiling", label: "Profiling dataset" },
    { key: "planning_agent", label: "AI planning analysis" },
    { key: "statistics", label: "Computing statistics" },
    { key: "visualization", label: "Generating charts" },
    { key: "insight_agent", label: "Writing insights" },
    { key: "critic_agent", label: "AI reviewing analysis" },
    { key: "report_generator", label: "Building PDF report" },
];

// ---------------------------------------------------------------------------
// Grab all the elements we'll need to update
// ---------------------------------------------------------------------------
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadError = document.getElementById("upload-error");

const uploadSection = document.getElementById("upload-section");
const progressSection = document.getElementById("progress-section");
const resultsSection = document.getElementById("results-section");

const progressBarInner = document.getElementById("progress-bar-inner");
const progressPercentText = document.getElementById("progress-percent-text");
const pipelineStagesList = document.getElementById("pipeline-stages");
const statusMessage = document.getElementById("status-message");

const downloadReportBtn = document.getElementById("download-report-btn");
const downloadDataBtn = document.getElementById("download-data-btn");
const startOverBtn = document.getElementById("start-over-btn");

let selectedFile = null;
let pollTimer = null;
let consecutiveFailures = 0;
const MAX_CONSECUTIVE_FAILURES = 5;
let currentJobId = null;

// ---------------------------------------------------------------------------
// Build the pipeline stage checklist once, on page load
// ---------------------------------------------------------------------------
function renderStageList() {
    pipelineStagesList.innerHTML = "";
    PIPELINE_STAGES.forEach(stage => {
        const li = document.createElement("li");
        li.id = `stage-${stage.key}`;
        li.innerHTML = `<span class="stage-icon"></span> ${stage.label}`;
        pipelineStagesList.appendChild(li);
    });
}
renderStageList();

// ---------------------------------------------------------------------------
// File selection — click to browse OR drag and drop
// ---------------------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFileSelected(fileInput.files[0]);
    }
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFileSelected(e.dataTransfer.files[0]);
    }
});

function handleFileSelected(file) {
    const allowedExtensions = [".csv", ".xlsx", ".xls"];
    const extension = "." + file.name.split(".").pop().toLowerCase();

    if (!allowedExtensions.includes(extension)) {
        uploadError.textContent = `Unsupported file type '${extension}'. Please upload .csv, .xlsx, or .xls`;
        selectedFile = null;
        uploadBtn.disabled = true;
        return;
    }

    uploadError.textContent = "";
    selectedFile = file;
    dropZoneText.textContent = `Selected: ${file.name}`;
    uploadBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Upload — calls POST /upload, then starts polling /status/{job_id}
// ---------------------------------------------------------------------------
uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";
    uploadError.textContent = "";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Upload failed.");
        }

        const data = await response.json();
        const jobId = data.job_id;

        showProgressSection();
        startPolling(jobId);

    } catch (err) {
        uploadError.textContent = err.message;
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Analyze Dataset";
    }
});

// ---------------------------------------------------------------------------
// Polling — checks GET /status/{job_id} every 2 seconds until done
// ---------------------------------------------------------------------------
function startPolling(jobId) {
    consecutiveFailures = 0;
    pollTimer = setInterval(() => checkStatus(jobId), 2000);
    checkStatus(jobId); // check immediately too, don't wait 2s for the first check
}

async function checkStatus(jobId) {
    try {
        const response = await fetch(`${API_BASE}/status/${jobId}`);
        if (!response.ok) {
            throw new Error("Could not fetch job status.");
        }
        const status = await response.json();
        consecutiveFailures = 0; // reset on any success
        updateProgressUI(status);

        if (status.status === "completed") {
            clearInterval(pollTimer);
            showResultsSection(jobId);
        } else if (status.status === "failed") {
            clearInterval(pollTimer);
            statusMessage.textContent = `Analysis failed: ${status.error || "Unknown error"}`;
            statusMessage.style.color = "#C0392B";
        }
    } catch (err) {
        consecutiveFailures++;
        // Only give up after several IN A ROW fail — a single hiccup shouldn't stop everything
        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            statusMessage.textContent = "Lost connection to the server. Please refresh and try again.";
            statusMessage.style.color = "#C0392B";
            clearInterval(pollTimer);
        }
        // otherwise, silently retry on the next 2-second tick
    }
}

function updateProgressUI(status) {
    progressBarInner.style.width = `${status.progress_percent}%`;
    progressPercentText.textContent = `${status.progress_percent}%`;

    const completed = status.completed_stages || [];
    const current = status.current_stage;

    PIPELINE_STAGES.forEach(stage => {
        const li = document.getElementById(`stage-${stage.key}`);
        if (!li) return;
        const wasAlreadyDone = li.classList.contains("done");
        li.classList.remove("active");

        if (completed.includes(stage.key)) {
            if (!wasAlreadyDone) {
                li.classList.add("done"); // triggers the checkmark pop animation via CSS
            }
        } else if (stage.key === current) {
            li.classList.add("active");
        }
    });

    if (status.revision_count > 0) {
        statusMessage.textContent = `AI is revising the analysis based on its own review (revision ${status.revision_count})...`;
    } else if (status.note) {
        statusMessage.textContent = status.note;
    } else {
        statusMessage.textContent = "";
    }
}

// ---------------------------------------------------------------------------
// Downloads — fetches as Blob and forces file save via URL.createObjectURL
// ---------------------------------------------------------------------------
downloadReportBtn.addEventListener("click", () => triggerDownload(currentJobId, "report", "pdf"));
downloadDataBtn.addEventListener("click", () => triggerDownload(currentJobId, "cleaned-data", "csv"));

async function triggerDownload(jobId, endpoint, extension) {
    if (!jobId) return;
    const btn = endpoint === "report" ? downloadReportBtn : downloadDataBtn;
    const originalText = btn.textContent;
    btn.innerHTML = `<span class="spinner"></span> Preparing download...`;
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/${endpoint}/${jobId}`);
        if (!response.ok) throw new Error("Download failed.");

        // Convert response into raw bytes (Blob) and trigger forced browser download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `autoinsight_${endpoint}_${jobId}.${extension}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

    } catch (err) {
        alert(`Download failed: ${err.message}`);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Section visibility helpers
// ---------------------------------------------------------------------------
function showProgressSection() {
    uploadSection.classList.add("hidden");
    progressSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
}

function showResultsSection(jobId) {
    currentJobId = jobId;
    progressSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");
}

startOverBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");

    selectedFile = null;
    currentJobId = null;
    consecutiveFailures = 0;
    fileInput.value = "";
    dropZoneText.textContent = "Drag & drop a file here, or click to browse";
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Analyze Dataset";

    // Reset stage visual state
    document.querySelectorAll(".pipeline-stages li").forEach(li => {
        li.classList.remove("done", "active");
    });
    progressBarInner.style.width = "0%";
    progressPercentText.textContent = "0%";
    statusMessage.textContent = "";
    statusMessage.style.color = "";
});