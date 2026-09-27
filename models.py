"""Local LLM integration: Ollama (CPU) + llama.cpp (GPU)."""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def build_cpu_model(model: str = "qwen3.8") -> ChatOllama:
    return ChatOllama(model=model)


def build_gpu_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        model="local",          # llama-server serves whatever model it loaded regardless of this string
        temperature=1.0,
    )
