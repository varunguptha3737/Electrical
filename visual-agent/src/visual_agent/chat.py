"""Plain vision-chat mode: converse with the VLM about uploaded images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_llm
from .tools.vision import image_block

CHAT_SYSTEM = SystemMessage(
    "You are a helpful visual assistant. You can analyze images, screenshots, "
    "documents, diagrams and charts the user shares, and answer questions "
    "about them precisely."
)


def build_user_message(text: str, images: list[str | Path] | None = None) -> HumanMessage:
    if not images:
        return HumanMessage(content=text)
    content: list = [{"type": "text", "text": text}]
    content += [image_block(p) for p in images]
    return HumanMessage(content=content)


def stream_chat(history: list, llm=None) -> Iterator[str]:
    """Stream the assistant reply for a message history (no system msg needed)."""
    llm = llm or get_llm()
    for chunk in llm.stream([CHAT_SYSTEM] + history):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content
