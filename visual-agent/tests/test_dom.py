"""Set-of-Marks grounding tests against the local test page (headless Chromium)."""

import io
from pathlib import Path

import pytest
from PIL import Image

from visual_agent.tools.browser import BrowserSession
from visual_agent.tools.dom import annotate_screenshot, elements_summary, extract_elements

PAGE = (Path(__file__).parent / "pages" / "test_page.html").resolve().as_uri()


@pytest.fixture(scope="module")
def session():
    s = BrowserSession(headless=True).start()
    s.navigate(PAGE)
    yield s
    s.stop()


def test_extract_elements_finds_button_and_input(session):
    elements = session.refresh_elements()
    by_role = {e.role: e for e in elements}
    assert "button" in by_role
    assert by_role["button"].name == "Click me"
    # button is at (100,100) size 200x100 → center ~ (200,150)
    cx, cy = by_role["button"].center
    assert abs(cx - 200) <= 3 and abs(cy - 150) <= 3
    assert any(e.role.startswith("input") for e in elements)
    summary = elements_summary(elements)
    assert '"Click me"' in summary and summary.startswith("Interactive elements")


def test_click_and_type_by_element_id(session):
    elements = session.refresh_elements()
    button = next(e for e in elements if e.role == "button")
    field = next(e for e in elements if e.role.startswith("input"))

    session.click_element(button.element_id)
    assert session.page.locator("#result").text_content() == "button-clicked"

    session.type_in_element(field.element_id, "som works")
    assert session.page.locator("#name-input").input_value() == "som works"

    with pytest.raises(ValueError, match="No element"):
        session.click_element(999)


def test_annotate_screenshot_draws_marks(session):
    png = session.screenshot()
    elements = session.refresh_elements()
    annotated = annotate_screenshot(png, elements)
    img = Image.open(io.BytesIO(annotated))
    assert img.size == (session.width, session.height)
    # the red mark color must appear in the annotated image
    colors = {c for _, c in img.convert("RGB").getcolors(maxcolors=1_000_000)}
    assert (255, 60, 60) in colors
