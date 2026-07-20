from app.graph import build_graph

g = build_graph(db_path=":memory:")
result = g.invoke(
    {"file_path": "sample_data/categorical_sample.csv"},
    config={"configurable": {"thread_id": "manual-day8"}},
)

print("Revision count:", result["revision_count"])
print("Final critic feedback:", result["critic_feedback"])
print("\nFinal plan analyses:", result["plan"]["analyses"])
exit()