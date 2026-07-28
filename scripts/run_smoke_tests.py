"""
Runs the full pipeline against several genuinely different real-world
datasets and reports pass/fail for each, plus a LangSmith trace link
where possible. Run this any time you make a significant change, to
catch bugs that only show up on messy/varied data — not just your
clean, hand-picked unit test fixtures.
"""
import time
import uuid
from app.graph import build_graph, run_pipeline_safely, PipelineExecutionError
from app.config import config

DATASETS = [
    ("Survey (Likert/categorical)", "sample_data/survey_sample.csv"),
    ("Heavy Nulls (messy)", "sample_data/heavy_nulls_sample.csv"),
    ("Time Series", "sample_data/timeseries_sample.csv"),
]


def run_smoke_tests():
    graph = build_graph()
    results = []

    for name, path in DATASETS:
        thread_id = f"smoke-{uuid.uuid4().hex[:8]}"
        start = time.time()
        print(f"\n{'='*60}\nRunning: {name} ({path})\n{'='*60}")

        try:
            result = run_pipeline_safely(
                graph, {"file_path": path},
                {"configurable": {"thread_id": thread_id}},
            )
            elapsed = time.time() - start
            report_path = result.get("report_path")
            revision_count = result.get("revision_count", 0)

            print(f"✅ PASSED in {elapsed:.1f}s — report: {report_path}, revisions: {revision_count}")
            results.append({"name": name, "status": "PASS", "time": elapsed,
                             "thread_id": thread_id, "revisions": revision_count})

        except PipelineExecutionError as e:
            elapsed = time.time() - start
            print(f"❌ FAILED in {elapsed:.1f}s — stage: {e.stage}, error: {e.message}")
            results.append({"name": name, "status": "FAIL", "time": elapsed,
                             "thread_id": thread_id, "stage": e.stage, "error": e.message})

    print(f"\n{'='*60}\nSMOKE TEST SUMMARY\n{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"{passed}/{len(results)} datasets passed\n")

    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{icon} {r['name']} — thread_id: {r['thread_id']}")
        if r["status"] == "FAIL":
            print(f"   Failed at: {r['stage']} — {r['error']}")

    if config.LANGCHAIN_TRACING_V2 == "true":
        print(f"\nView traces at: https://smith.langchain.com (project: {config.LANGCHAIN_PROJECT})")
        print("Search by thread_id (shown above) to find each run's full trace.")

    return results


if __name__ == "__main__":
    run_smoke_tests()