from app.graph import build_graph
g = build_graph(db_path=":memory:")
result = g.invoke(
    {"file_path": "sample_data/Sales-Export_2019-2020.csv"},
    config={"configurable": {"thread_id": "sales-retest"}},
)
print("Report:", result["report_path"])
print("Numeric columns found:", result["profile"]["numeric_columns"])
exit()