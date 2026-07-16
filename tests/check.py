import time
from app.graph import build_graph

g = build_graph(db_path=":memory:")
start = time.time()
result = g.invoke(
    {"file_path": "sample_data/timeseries_sample.csv"},
    config={"configurable": {"thread_id": "timing-test"}},
)
print(f"Total time: {time.time() - start:.2f}s")
print("Statistics present:", "statistics" in result)
print("Visualizations present:", "visualizations" in result)
exit()