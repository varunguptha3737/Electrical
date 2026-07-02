"""Image loading/encoding helpers for multimodal messages."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from ..config import settings


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def encode_image(
    source: str | Path | bytes | Image.Image, max_edge: int | None = None
) -> str:
    """Return a base64 PNG data URL, downscaled so the longest edge fits max_edge."""
    max_edge = max_edge or settings.max_image_edge
    img = load_image(source)
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)),
            Image.LANCZOS,
        )
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def image_block(source: str | Path | bytes | Image.Image, max_edge: int | None = None) -> dict:
    """OpenAI-style image content block for a chat message."""
    return {"type": "image_url", "image_url": {"url": encode_image(source, max_edge)}}
