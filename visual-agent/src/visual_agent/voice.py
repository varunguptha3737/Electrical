"""Voice mode: full-duplex speech conversation (ChatGPT-voice style).

Pipeline:  browser mic → FastRTC WebRTC (Silero VAD detects when you stop
talking; can_interrupt lets you barge in mid-reply) → STT → intent routing →
either streamed chat (spoken sentence-by-sentence as tokens arrive) or a
computer-use task (with short spoken progress narration) → Kokoro TTS.

All models load lazily on first use so importing this module is cheap.
"""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage

from .chat import stream_chat
from .config import settings
from .llm import get_llm

_stt_model = None
_tts_model = None


# --- Speech-to-text ---------------------------------------------------------


def _to_float_mono_16k(audio: tuple[int, np.ndarray]) -> np.ndarray:
    sr, data = audio
    data = np.asarray(data)
    if data.ndim > 1:  # (channels, n) or (n, channels) -> mono
        data = data.mean(axis=0 if data.shape[0] < data.shape[1] else 1)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n), np.arange(len(data)), data
        ).astype(np.float32)
    return data


def get_stt():
    """Lazy-load the STT backend (faster-whisper by default)."""
    global _stt_model
    if _stt_model is not None:
        return _stt_model
    if settings.stt_backend == "moonshine":
        from fastrtc import get_stt_model

        model = get_stt_model()
        _stt_model = ("moonshine", model)
    else:
        from faster_whisper import WhisperModel

        device = settings.whisper_device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
        _stt_model = ("whisper", model)
    return _stt_model


def transcribe(audio: tuple[int, np.ndarray]) -> str:
    kind, model = get_stt()
    if kind == "moonshine":
        return model.stt(audio).strip()
    segments, _info = model.transcribe(_to_float_mono_16k(audio), beam_size=1)
    return " ".join(seg.text for seg in segments).strip()


# --- Text-to-speech ---------------------------------------------------------


def get_tts():
    """Lazy-load Kokoro TTS via FastRTC."""
    global _tts_model
    if _tts_model is None:
        from fastrtc import get_tts_model

        _tts_model = get_tts_model()  # Kokoro-82M
    return _tts_model


def speak(text: str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield streamed audio chunks for a piece of text."""
    text = text.strip()
    if not text:
        return
    yield from get_tts().stream_tts_sync(text)


# --- Streaming sentence chunker ---------------------------------------------

_SENTENCE_END = re.compile(r"([.!?։。？！]|\n)+[\s\"')\]]*$")


def sentence_chunks(tokens: Iterator[str], min_len: int = 25) -> Iterator[str]:
    """Group a token stream into sentence-ish chunks so TTS can start speaking
    while the LLM is still generating (low latency + clean barge-in points)."""
    buf = ""
    for token in tokens:
        buf += token
        if len(buf) >= min_len and _SENTENCE_END.search(buf):
            yield buf.strip()
            buf = ""
    if buf.strip():
        yield buf.strip()


# --- Intent routing ----------------------------------------------------------

ROUTER_PROMPT = """Classify the user's spoken request into exactly one word:
- "browse" if it asks to operate a computer/web browser, open a website or
  application, search the web, click/fill/do something on screen.
- "chat" for anything else (questions, conversation, analysis).
Request: {text}
Answer with only the single word browse or chat."""


def route_intent(text: str, llm=None) -> str:
    llm = llm or get_llm()
    reply = llm.invoke(ROUTER_PROMPT.format(text=text))
    content = reply.content if hasattr(reply, "content") else str(reply)
    return "browse" if "browse" in str(content).lower() else "chat"


def narrate_action(tool_name: str, tool_args: dict) -> str | None:
    """Short spoken progress line for a computer-use step."""
    if tool_name == "navigate":
        return f"Opening {tool_args.get('url', 'the page')}."
    if tool_name == "open_app":
        return f"Launching {tool_args.get('command', 'the application')}."
    if tool_name in ("click_element", "click"):
        return "Clicking."
    if tool_name in ("type_in_element", "type_text"):
        return f"Typing {tool_args.get('text', '')}."
    if tool_name == "scroll":
        return "Scrolling."
    return None  # press_key / go_back / finish: stay quiet


# --- FastRTC handler ---------------------------------------------------------


class VoiceAssistant:
    """ReplyOnPause handler: one call per detected user utterance.

    Yields (sample_rate, ndarray) audio chunks. FastRTC stops consuming the
    generator when the user interrupts (barge-in), which also cancels the
    in-flight LLM stream.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.history: list = []

    def __call__(self, audio: tuple[int, np.ndarray]):
        text = transcribe(audio)
        if not text:
            return
        llm = self.llm or get_llm()
        if route_intent(text, llm) == "browse":
            yield from self._run_task(text, llm)
        else:
            yield from self._chat(text, llm)

    def _chat(self, text: str, llm):
        self.history.append(HumanMessage(content=text))
        reply = ""
        for sentence in sentence_chunks(stream_chat(self.history, llm=llm, mode="voice")):
            reply += sentence + " "
            yield from speak(sentence)
        self.history.append(AIMessage(content=reply.strip()))

    def _run_task(self, text: str, llm):
        from .agent import run_computer_task

        yield from speak("Okay, on it.")
        final = None
        for event in run_computer_task(text, llm=llm):
            if event.kind == "tool" and settings.voice_narration:
                line = narrate_action(event.tool_name, event.tool_args)
                if line:
                    yield from speak(line)
            elif event.kind == "final":
                final = event.text
        self.history.append(HumanMessage(content=text))
        self.history.append(AIMessage(content=final or "Task finished."))
        from .memory import after_exchange

        after_exchange(text, final or "Task finished.", llm, mode="voice-task")
        yield from speak(final or "The task is finished.")
