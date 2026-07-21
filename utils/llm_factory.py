import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

def get_llm(temperature: float = 0.2):
    """Returns the appropriate LLM instance based on the LLM_PROVIDER environment variable."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )
    elif provider == "groq":
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(
            model=model_name,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")