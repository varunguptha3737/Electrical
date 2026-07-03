"""Plain vision-chat mode: converse with the VLM about uploaded images.

Long-term memory is woven in transparently: relevant notes from the markdown
vault are injected into the system prompt, and after each exchange the model
decides (in a background thread) whether it learned something worth saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .llm import get_llm
from .tools.vision import image_block

CHAT_SYSTEM = (
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


def _text_of(message) -> str:
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        b.get("text", "") for b in message.content if isinstance(b, dict)
    ).strip()


def chat_system(query: str = "") -> SystemMessage:
    text = CHAT_SYSTEM
    if settings.memory_enabled:
        from .memory import get_vault

        memory = get_vault().context(query)
        if memory:
            text += (
                "\n\nThings you remember about the user from earlier "
                "conversations (use them naturally, don't recite them):\n" + memory
            )
    return SystemMessage(text)


def stream_chat(history: list, llm=None, mode: str = "chat") -> Iterator[str]:
    """Stream the assistant reply for a message history (no system msg needed)."""
    llm = llm or get_llm()
    user_text = _text_of(history[-1]) if history else ""
    system = chat_system(user_text)
    reply = ""
    try:
        for chunk in llm.stream([system] + history):
            if isinstance(chunk.content, str) and chunk.content:
                reply += chunk.content
                yield chunk.content
    finally:
        # runs even when the stream is cut short (voice barge-in)
        from .memory import after_exchange

        after_exchange(user_text, reply, llm, mode=mode)
