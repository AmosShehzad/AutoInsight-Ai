from app.graph import build_graph

g = build_graph()

result = g.invoke({"file_path": "sample_data/clean_sample.csv"})
print(result["validation_report"]["health_score"])   # should be 100.0

result2 = g.invoke({"file_path": "sample_data/messy_sample.csv"})
print(result2["cleaning_report"])   # should show duplicates_removed, missing_value_handling
print(result2["profile"]["numeric_stats"])