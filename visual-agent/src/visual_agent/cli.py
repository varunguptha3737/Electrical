"""Command-line interface: chat / browse / ui."""

from __future__ import annotations

from pathlib import Path

import typer
from langchain_core.messages import AIMessage

from .config import settings

app = typer.Typer(
    name="visual-agent",
    help="Visual agentic AI on a local open-source VLM (vision chat + browser computer-use).",
    no_args_is_help=True,
)


@app.command()
def chat(
    image: list[Path] = typer.Option(
        None, "--image", "-i", help="Image file(s) to attach to your first message."
    ),
):
    """Interactive vision chat with the local model."""
    from .chat import build_user_message, stream_chat

    history: list = []
    pending_images = [str(p) for p in image] if image else []
    typer.echo(f"Model: {settings.model_name} @ {settings.vllm_base_url}")
    typer.echo("Type your message ('exit' to quit).\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"exit", "quit"}:
            break
        history.append(build_user_message(text, pending_images))
        pending_images = []
        typer.echo("agent> ", nl=False)
        reply = ""
        for token in stream_chat(history):
            reply += token
            typer.echo(token, nl=False)
        typer.echo("\n")
        history.append(AIMessage(content=reply))


def _print_events(events):
    for event in events:
        if event.kind == "thought":
            typer.secho(f"[thought] {event.text}", fg=typer.colors.CYAN)
        elif event.kind == "tool":
            typer.secho(f"[action]  {event.tool_name}({event.tool_args})", fg=typer.colors.YELLOW)
        elif event.kind == "screenshot":
            typer.echo(f"[shot]    {event.screenshot_path}")
        elif event.kind == "final":
            typer.secho(f"\n[result]  {event.text}", fg=typer.colors.GREEN, bold=True)


@app.command()
def browse(
    task: str = typer.Argument(..., help="Natural-language task for the browser agent."),
    max_steps: int = typer.Option(None, help=f"Step limit (default {settings.max_steps})."),
):
    """Run an autonomous browser computer-use task."""
    from .agent import run_browse_task

    typer.echo(f"Model: {settings.model_name} @ {settings.vllm_base_url}")
    typer.echo(f"Task: {task}\n")
    _print_events(run_browse_task(task, max_steps=max_steps))


@app.command()
def desktop(
    task: str = typer.Argument(..., help="Task for the OS-level desktop agent."),
    max_steps: int = typer.Option(None, help=f"Step limit (default {settings.max_steps})."),
):
    """Control the WHOLE desktop: open apps (LibreOffice, GIMP…), click, type.

    Needs a display and the desktop extra: pip install -e ".[desktop]"
    """
    from .agent import run_desktop_task

    typer.secho(
        "⚠ The agent will control your real mouse/keyboard. "
        "Slam the mouse into a screen corner to abort.",
        fg=typer.colors.RED,
    )
    typer.echo(f"Model: {settings.model_name} @ {settings.vllm_base_url}")
    typer.echo(f"Task: {task}\n")
    _print_events(run_desktop_task(task, max_steps=max_steps))


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(7860, help="Port."),
):
    """Launch the Gradio web UI."""
    from .ui import build_ui

    build_ui().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    app()
