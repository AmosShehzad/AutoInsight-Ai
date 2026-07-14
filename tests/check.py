from app.graph import build_graph

g = build_graph(db_path=":memory:")
result = g.invoke(
    {"file_path": "sample_data/timeseries_sample.csv"},
    config={"configurable": {"thread_id": "manual-day5"}},
)

print("Requested analyses:", result["statistics"]["requested_analyses"])
print("Unsupported (if any):", result["statistics"]["unsupported_analyses"])
print("\nResults:")
for name, data in result["statistics"]["results"].items():
    print(f"\n{name}: {data}")
exit()