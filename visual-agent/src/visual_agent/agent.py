"""LangGraph agent: ReAct loop over screenshots + browser tools.

Flow:  agent (VLM picks a tool) -> tools (execute in Playwright, capture a
fresh screenshot, prune old ones from context) -> agent ... until the model
calls `finish` or the step budget runs out.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from .config import settings
from .llm import get_llm
from .tools.browser import BrowserSession, make_browser_tools
from .tools.dom import annotate_screenshot, elements_summary
from .tools.vision import image_block

SYSTEM_PROMPT = """You are a computer-use agent controlling a web browser \
through screenshots. The browser viewport is {width}x{height} pixels.

Rules:
- After every action you receive a fresh screenshot of the current page. \
Always ground your next action on the LATEST screenshot.
{grounding_rules}
- If the page looks wrong or an action had no effect, scroll or try a \
slightly different coordinate rather than repeating the same click.
- Work step by step: ONE tool call per turn. Briefly state what you see and \
what you are doing before each call.
- When the task is done (or clearly impossible), call `finish` with a \
complete answer for the user. Never call finish before actually completing \
the task."""

PIXEL_RULES = """\
- To click, give exact pixel coordinates of the center of the target element \
as seen in the latest screenshot (x from left, y from top).
- To fill a text field: click it first, then use type_text, then press_key \
'Enter' if a submission is needed."""

HYBRID_RULES = """\
- Interactive elements are numbered with red labels on the screenshot and \
listed as text. Use click_element(id) and type_in_element(id, text) with \
those numbers — they always hit exactly. Only fall back to pixel click(x, y) \
for unnumbered targets (maps, canvases, sliders).
- After typing in a search box, press_key 'Enter' to submit."""


class AgentState(TypedDict):
    messages: list
    steps: int
    final_answer: str | None


@dataclass
class StepEvent:
    """One unit of agent progress, streamed to the CLI / web UI."""

    kind: str  # "thought" | "tool" | "screenshot" | "final"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    screenshot_path: Path | None = None


def prune_screenshots(messages: list, keep: int) -> list:
    """Replace all but the last `keep` screenshot messages with a stub so the
    context doesn't fill up with stale images."""
    image_idx = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, HumanMessage)
        and isinstance(m.content, list)
        and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m.content)
    ]
    pruned = list(messages)
    for i in image_idx[:-keep] if keep > 0 else image_idx:
        pruned[i] = HumanMessage(content="[earlier screenshot removed to save context]")
    return pruned


def build_task_graph(llm, tools: list, observe, system: SystemMessage,
                     run_dir: Path, max_steps: int):
    """Generic screenshot-act loop. `observe()` returns (png_bytes, caption)
    for the post-action observation — browser and desktop supply their own."""
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke([system] + state["messages"])
        return {
            "messages": state["messages"] + [response],
            "steps": state["steps"] + 1,
            "final_answer": None,
        }

    def tools_node(state: AgentState) -> AgentState:
        ai: AIMessage = state["messages"][-1]
        messages = list(state["messages"])
        for tc in ai.tool_calls:
            try:
                result = tools_by_name[tc["name"]].invoke(tc["args"])
            except Exception as exc:  # surface tool failures to the model
                result = f"ERROR: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        png, caption = observe()
        shot_path = run_dir / f"step_{state['steps']:03d}.png"
        shot_path.write_bytes(png)
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": caption},
                    image_block(png),
                ]
            )
        )
        messages = prune_screenshots(messages, keep=settings.max_screenshots)
        return {"messages": messages, "steps": state["steps"], "final_answer": None}

    def finalize_node(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        messages = list(state["messages"])
        answer = None
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                if tc["name"] == "finish":
                    answer = tc["args"].get("answer", "")
                messages.append(
                    ToolMessage(content=answer or "done", tool_call_id=tc["id"])
                )
        if answer is None:
            answer = last.content if isinstance(last.content, str) else str(last.content)
            if state["steps"] >= max_steps:
                answer = (
                    f"[Stopped: reached the {max_steps}-step limit] "
                    f"Last model output: {answer}"
                )
        return {"messages": messages, "steps": state["steps"], "final_answer": answer}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return "finalize"
        if any(tc["name"] == "finish" for tc in last.tool_calls):
            return "finalize"
        if state["steps"] >= max_steps:
            return "finalize"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def new_run_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(settings.runs_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _stream_graph(graph, task: str, run_dir: Path, max_steps: int) -> Iterator[StepEvent]:
    initial: AgentState = {
            "messages": [HumanMessage(content=f"Task: {task}")],
        "steps": 0,
        "final_answer": None,
    }
    seen = 0
    for update in graph.stream(
        initial, {"recursion_limit": max_steps * 2 + 10}, stream_mode="updates"
    ):
        for node, state in update.items():
            if node == "agent":
                ai: AIMessage = state["messages"][-1]
                if isinstance(ai.content, str) and ai.content.strip():
                    yield StepEvent(kind="thought", text=ai.content.strip())
                for tc in ai.tool_calls:
                    yield StepEvent(
                        kind="tool", tool_name=tc["name"], tool_args=tc["args"]
                    )
            elif node == "tools":
                shot = run_dir / f"step_{state['steps']:03d}.png"
                if shot.exists() and state["steps"] > seen:
                    seen = state["steps"]
                    yield StepEvent(kind="screenshot", screenshot_path=shot)
            elif node == "finalize":
                yield StepEvent(kind="final", text=state["final_answer"] or "")


def run_browse_task(
    task: str,
    llm=None,
    session: BrowserSession | None = None,
    max_steps: int | None = None,
) -> Iterator[StepEvent]:
    """Run a browser computer-use task, yielding StepEvents as the agent works."""
    llm = llm or get_llm()
    max_steps = max_steps or settings.max_steps
    run_dir = new_run_dir()
    hybrid = settings.agent_vision_mode == "hybrid"

    own_session = session is None
    if own_session:
        session = BrowserSession().start()

    def observe() -> tuple[bytes, str]:
        png = session.screenshot()
        if not hybrid:
            return png, "Current browser screenshot:"
        elements = session.refresh_elements()
        return (
            annotate_screenshot(png, elements),
            "Current browser screenshot (interactive elements numbered):\n"
            + elements_summary(elements),
        )

    system = SystemMessage(
        SYSTEM_PROMPT.format(
            width=session.width,
            height=session.height,
            grounding_rules=HYBRID_RULES if hybrid else PIXEL_RULES,
        )
    )
    try:
        graph = build_task_graph(
            llm, make_browser_tools(session), observe, system, run_dir, max_steps
        )
        yield from _stream_graph(graph, task, run_dir, max_steps)
    finally:
        if own_session:
            session.stop()


def run_desktop_task(
    task: str,
    llm=None,
    session=None,
    max_steps: int | None = None,
) -> Iterator[StepEvent]:
    """Run an OS-level computer-use task (whole desktop: open apps, edit
    files/photos on screen). Requires a display — see README."""
    from .tools.desktop import DESKTOP_SYSTEM_RULES, DesktopSession, make_desktop_tools

    llm = llm or get_llm()
    max_steps = max_steps or settings.max_steps
    run_dir = new_run_dir()

    own_session = session is None
    if own_session:
        session = DesktopSession().start()

    def observe() -> tuple[bytes, str]:
        return session.screenshot(), "Current screen:"

    system = SystemMessage(
        SYSTEM_PROMPT.format(
            width=session.width,
            height=session.height,
            grounding_rules=DESKTOP_SYSTEM_RULES,
        ).replace("a web browser", "a computer desktop")
    )
    try:
        graph = build_task_graph(
            llm, make_desktop_tools(session), observe, system, run_dir, max_steps
        )
        yield from _stream_graph(graph, task, run_dir, max_steps)
    finally:
        if own_session:
            session.stop()


def run_computer_task(task: str, **kwargs) -> Iterator[StepEvent]:
    """Dispatch to browser or desktop control based on COMPUTER_MODE."""
    if settings.computer_mode == "desktop":
        yield from run_desktop_task(task, **kwargs)
    else:
        yield from run_browse_task(task, **kwargs)
