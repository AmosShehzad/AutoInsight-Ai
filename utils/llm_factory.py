"""
Central LLM factory — every agent (Planning, Insight, Critic) builds its
ChatGroq instance through this ONE function, instead of each file
constructing ChatGroq(...) independently. This is what makes it possible
to change the LLM provider/model in exactly one place later.
"""
"""
Central LLM factory — every agent builds its ChatGroq instance through
this ONE function. Tags let LangSmith's dashboard filter traces by
which agent made the call — e.g. "show me only Critic Agent runs".
"""
from langchain_groq import ChatGroq
from app.config import config


def get_llm(model: str = None, temperature: float = 0.2, tags: list[str] = None):
    """
    Returns a configured ChatGroq instance.
    tags: list of strings attached to every call this LLM instance makes —
    shows up in LangSmith so you can filter traces by agent name.
    """
    return ChatGroq(
        model=model or config.LLM_MODEL,
        temperature=temperature,
        tags=tags or [],
    )