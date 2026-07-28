"""
Generates a visual diagram of the actual LangGraph pipeline structure —
auto-drawn from your real graph.py, not hand-drawn. Run this once and
commit the output image into your README.
"""
from app.graph import build_graph

graph = build_graph()

# LangGraph's built-in Mermaid diagram generator
png_bytes = graph.get_graph().draw_mermaid_png()

with open("docs/graph_diagram.png", "wb") as f:
    f.write(png_bytes)

print("Diagram saved to docs/graph_diagram.png")