from app.graph import build_graph

g = build_graph(db_path=":memory:")

for name, path in [
    ("TIME SERIES", "sample_data/timeseries_sample.csv"),
    ("CATEGORICAL", "sample_data/categorical_sample.csv"),
    ("OUTLIER", "sample_data/outlier_sample.csv"),
]:
    result = g.invoke({"file_path": path}, config={"configurable": {"thread_id": f"manual-{name}"}})
    print(f"\n=== {name} ===")
    for insight in result["insights"]:
        print(f"  - {insight}")
exit()