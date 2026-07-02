"""End-to-end agent-loop test with a scripted fake LLM and a real (headless)
Playwright browser against a local HTML page. No GPU / model server needed."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from visual_agent.agent import prune_screenshots, run_browse_task
from visual_agent.tools.browser import BrowserSession

PAGE = (Path(__file__).parent / "pages" / "test_page.html").resolve().as_uri()


class FakeToolCallingLLM:
    """Pops one scripted AIMessage per invoke; bind_tools is a no-op."""

    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


def _ai(name, args, call_id):
    return AIMessage(
        content=f"calling {name}",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture(scope="module")
def session():
    s = BrowserSession(headless=True).start()
    yield s
    s.stop()


def test_full_browse_loop(session, tmp_path, monkeypatch):
    monkeypatch.setattr("visual_agent.agent.settings.runs_dir", str(tmp_path))
    llm = FakeToolCallingLLM(
        [
            _ai("navigate", {"url": PAGE}, "c1"),
            # button spans (100,100)-(300,200): click its center
            _ai("click", {"x": 200, "y": 150}, "c2"),
            # focus the input, then type
            _ai("click", {"x": 200, "y": 270}, "c3"),
            _ai("type_text", {"text": "hello agent"}, "c4"),
            _ai("finish", {"answer": "clicked and typed"}, "c5"),
        ]
    )

    events = list(run_browse_task("test task", llm=llm, session=session, max_steps=10))

    kinds = [e.kind for e in events]
    assert kinds.count("tool") == 5
    assert kinds[-1] == "final"
    assert events[-1].text == "clicked and typed"

    # the browser really did what the fake model asked
    assert session.page.locator("#result").text_content() == "button-clicked"
    assert session.page.locator("#name-input").input_value() == "hello agent"

    # step screenshots were captured for each acting step (not the finish)
    shots = [e for e in events if e.kind == "screenshot"]
    assert len(shots) == 4
    assert all(e.screenshot_path.exists() for e in shots)


def test_max_steps_cutoff(session, tmp_path, monkeypatch):
    monkeypatch.setattr("visual_agent.agent.settings.runs_dir", str(tmp_path))
    llm = FakeToolCallingLLM(
        [
            _ai("navigate", {"url": PAGE}, "s1"),
            _ai("scroll", {"direction": "down", "amount": 100}, "s2"),
            _ai("scroll", {"direction": "down", "amount": 100}, "s3"),
        ]
    )
    events = list(run_browse_task("loop forever", llm=llm, session=session, max_steps=3))
    assert events[-1].kind == "final"
    assert "3-step limit" in events[-1].text


def test_prune_screenshots_keeps_recent():
    def shot(i):
        return HumanMessage(
            content=[
                {"type": "text", "text": f"shot {i}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ]
        )

    msgs = [HumanMessage(content="task"), shot(1), shot(2), shot(3), shot(4)]
    pruned = prune_screenshots(msgs, keep=2)

    def has_image(m):
        return isinstance(m.content, list) and any(
            b.get("type") == "image_url" for b in m.content if isinstance(b, dict)
        )

    assert [has_image(m) for m in pruned] == [False, False, False, True, True]
    assert pruned[0].content == "task"  # non-image messages untouched
