"""Gradio web UI: vision chat tab + computer-use tab with live screenshots."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from langchain_core.messages import AIMessage

from .chat import build_user_message, stream_chat
from .config import settings


def _latest_run_screenshot() -> str | None:
    shots = sorted(Path(settings.runs_dir).glob("*/step_*.png"))
    return str(shots[-1]) if shots else None


def _chat_fn(message: str, image, history_state: list, chatbot: list):
    """Stream a vision-chat reply; `history_state` holds LangChain messages."""
    if not message.strip() and image is None:
        yield chatbot, history_state, gr.update()
        return
    images = [image] if image is not None else None
    history_state = history_state + [build_user_message(message, images)]
    chatbot = chatbot + [
        {"role": "user", "content": message + (" 🖼️ (image attached)" if images else "")},
        {"role": "assistant", "content": ""},
    ]
    reply = ""
    for token in stream_chat(history_state):
        reply += token
        chatbot[-1]["content"] = reply
        yield chatbot, history_state, gr.update(value=None)
    history_state.append(AIMessage(content=reply))
    yield chatbot, history_state, gr.update(value=None)


def _browse_fn(task: str, max_steps: int):
    """Run the computer-use agent, streaming the step log and screenshots."""
    from .agent import run_browse_task

    if not task.strip():
        yield [], None
        return
    log: list[dict] = [{"role": "user", "content": task}]
    screenshot = None
    yield log, screenshot
    try:
        for event in run_browse_task(task, max_steps=int(max_steps) or None):
            if event.kind == "thought":
                log.append({"role": "assistant", "content": f"💭 {event.text}"})
            elif event.kind == "tool":
                log.append(
                    {"role": "assistant", "content": f"🛠️ `{event.tool_name}({event.tool_args})`"}
                )
            elif event.kind == "screenshot":
                screenshot = str(event.screenshot_path)
            elif event.kind == "final":
                log.append({"role": "assistant", "content": f"✅ **{event.text}**"})
            yield log, screenshot
    except Exception as exc:
        log.append({"role": "assistant", "content": f"❌ Error: {exc}"})
        yield log, screenshot


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Visual Agent") as demo:
        gr.Markdown(
            f"# 👁️ Visual Agent\n"
            f"Local open-source VLM: `{settings.model_name}` @ `{settings.vllm_base_url}`"
        )

        with gr.Tab("Vision chat"):
            chatbot = gr.Chatbot(type="messages", height=480, label="Chat")
            history_state = gr.State([])
            with gr.Row():
                image_in = gr.Image(type="filepath", label="Attach image", scale=1)
                with gr.Column(scale=3):
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask about the image, or anything else…",
                    )
                    send = gr.Button("Send", variant="primary")
            for trigger in (send.click, msg.submit):
                trigger(
                    _chat_fn,
                    inputs=[msg, image_in, history_state, chatbot],
                    outputs=[chatbot, history_state, image_in],
                ).then(lambda: "", outputs=msg)

        with gr.Tab("Computer use"):
            gr.Markdown(
                "Give the agent a task; it drives a real browser from "
                "screenshots. Steps stream on the left, the live screenshot on the right."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    task_in = gr.Textbox(
                        label="Task",
                        placeholder="e.g. Go to en.wikipedia.org and find the population of Iceland",
                    )
                    steps_in = gr.Slider(
                        3, 40, value=settings.max_steps, step=1, label="Max steps"
                    )
                    run_btn = gr.Button("Run task", variant="primary")
                    step_log = gr.Chatbot(type="messages", height=420, label="Agent steps")
                with gr.Column(scale=1):
                    screen = gr.Image(label="Browser view", height=540)
            run_btn.click(_browse_fn, inputs=[task_in, steps_in], outputs=[step_log, screen])
            task_in.submit(_browse_fn, inputs=[task_in, steps_in], outputs=[step_log, screen])

        with gr.Tab("Voice"):
            _build_voice_tab()

    return demo


def _build_voice_tab() -> None:
    """Full-duplex voice conversation (talk, get spoken replies, interrupt
    mid-sentence). Voice commands like 'open Wikipedia and…' run the
    computer-use agent with spoken progress."""
    try:
        from fastrtc import ReplyOnPause, WebRTC

        from .voice import VoiceAssistant
    except ImportError as exc:
        gr.Markdown(
            f"⚠️ Voice dependencies not installed (`{exc.name}`). "
            'Run `pip install -e ".[voice]"` and restart.'
        )
        return

    gr.Markdown(
        "🎙️ **Talk to the agent.** It answers with a natural voice — you can "
        "interrupt it mid-sentence, just like ChatGPT voice. Say things like "
        "*“open Wikipedia and find the population of Iceland”* to run the "
        "computer-use agent by voice; its screen appears below while it works."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
                label="Voice conversation",
            )
        with gr.Column(scale=1):
            voice_screen = gr.Image(label="Agent screen (during voice tasks)", height=420)
    audio.stream(
        ReplyOnPause(VoiceAssistant(), can_interrupt=True),
        inputs=[audio],
        outputs=[audio],
    )
    timer = gr.Timer(2)
    timer.tick(_latest_run_screenshot, outputs=voice_screen)


if __name__ == "__main__":
    build_ui().launch()
