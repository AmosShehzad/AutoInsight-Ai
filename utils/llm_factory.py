"""
Central LLM factory — every agent (Planning, Insight, Critic) builds its
ChatGroq instance through this ONE function, instead of each file
constructing ChatGroq(...) independently. This is what makes it possible
to change the LLM provider/model in exactly one place later.
"""
from langchain_groq import ChatGroq
from app.config import config


def get_llm(model: str = None, temperature: float = 0.2):
    """
    Returns a configured ChatGroq instance.
    model: defaults to config.LLM_MODEL if not passed.
    temperature: caller-specified per agent (Planning/Insight/Critic use
    different values), no default assumed here.
    """
    return ChatGroq(
        model=model or config.LLM_MODEL,
        temperature=temperature,
    )