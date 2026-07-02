"""LLM client factory pointing at the local vLLM OpenAI-compatible endpoint."""

from langchain_openai import ChatOpenAI

from .config import settings


def get_llm(**overrides) -> ChatOpenAI:
    params = dict(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        max_retries=1,
        timeout=300,
    )
    params.update(overrides)
    return ChatOpenAI(**params)
