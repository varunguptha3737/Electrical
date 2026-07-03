"""Playwright browser session and the computer-use tools exposed to the agent.

The agent never sees the DOM — it acts purely from screenshots, clicking at
pixel coordinates the vision model grounds on the latest screenshot. The
viewport is fixed so model coordinates map 1:1 to Playwright mouse events.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ..config import settings
from .dom import PageElement, extract_elements


class BrowserSession:
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        headless: bool | None = None,
    ):
        self.width = width or settings.viewport_width
        self.height = height or settings.viewport_height
        self.headless = settings.headless if headless is None else headless
        self._pw = None
        self._browser = None
        self.page = None
        self.elements: list[PageElement] = []

    def start(self) -> "BrowserSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            executable_path=settings.chromium_executable or None,
        )
        context = self._browser.new_context(
            viewport={"width": self.width, "height": self.height}
        )
        self.page = context.new_page()
        return self

    def stop(self) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        self._pw = self._browser = self.page = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def screenshot(self) -> bytes:
        return self.page.screenshot(type="png")

    def refresh_elements(self) -> list[PageElement]:
        self.elements = extract_elements(self.page)
        return self.elements

    def _find_element(self, element_id: int) -> PageElement:
        for el in self.elements:
            if el.element_id == element_id:
                return el
        raise ValueError(
            f"No element [{element_id}] on the current screen "
            f"(valid ids: 1..{len(self.elements)}). Look at the latest screenshot."
        )

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("load", timeout=10_000)
        except Exception:
            pass  # dynamic pages may never fire load; act on whatever rendered
        self.page.wait_for_timeout(400)

    # Actions -------------------------------------------------------------

    def navigate(self, url: str) -> str:
        if "://" not in url:
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self._settle()
        return f"Opened {url}. Page title: {self.page.title()!r}"

    def click(self, x: int, y: int) -> str:
        self.page.mouse.click(x, y)
        self._settle()
        return f"Clicked at ({x}, {y})."

    def type_text(self, text: str) -> str:
        self.page.keyboard.type(text, delay=20)
        return f"Typed {text!r}."

    def press_key(self, key: str) -> str:
        self.page.keyboard.press(key)
        self._settle()
        return f"Pressed {key!r}."

    def scroll(self, direction: str, amount: int = 500) -> str:
        dy = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, dy)
        self.page.wait_for_timeout(300)
        return f"Scrolled {direction} by {amount}px."

    def go_back(self) -> str:
        self.page.go_back(wait_until="domcontentloaded", timeout=30_000)
        self._settle()
        return f"Went back. Page title: {self.page.title()!r}"

    def click_element(self, element_id: int) -> str:
        el = self._find_element(element_id)
        x, y = el.center
        self.page.mouse.click(x, y)
        self._settle()
        return f"Clicked {el.describe()} at its center ({x}, {y})."

    def type_in_element(self, element_id: int, text: str) -> str:
        el = self._find_element(element_id)
        x, y = el.center
        self.page.mouse.click(x, y)
        self.page.keyboard.type(text, delay=20)
        return f"Typed {text!r} into {el.describe()}."


def make_browser_tools(session: BrowserSession) -> list:
    """Build the LangChain tools bound to one live browser session."""

    @tool
    def navigate(url: str) -> str:
        """Open a URL in the browser. Use a full URL like https://example.com."""
        return session.navigate(url)

    @tool
    def click_element(element_id: int) -> str:
        """Click an interactive element by the numbered id shown on the latest
        screenshot / element list. PREFER this over pixel clicks — it always
        hits the element exactly."""
        return session.click_element(element_id)

    @tool
    def type_in_element(element_id: int, text: str) -> str:
        """Click a numbered input element to focus it, then type the text
        into it. PREFER this for filling forms."""
        return session.type_in_element(element_id, text)

    @tool
    def click(x: int, y: int) -> str:
        """Click at pixel coordinates (x, y) on the current page. Only use
        this when the target has NO numbered id (e.g. a spot on a map or
        canvas). x is from the left edge, y from the top edge of the LATEST
        screenshot."""
        return session.click(x, y)

    @tool
    def type_text(text: str) -> str:
        """Type text with the keyboard into the currently focused element.
        Click the input field first to focus it."""
        return session.type_text(text)

    @tool
    def press_key(key: str) -> str:
        """Press a single key, e.g. 'Enter', 'Tab', 'Escape', 'ArrowDown',
        'PageDown'."""
        return session.press_key(key)

    @tool
    def scroll(direction: str, amount: int = 500) -> str:
        """Scroll the page 'up' or 'down' by the given number of pixels."""
        return session.scroll(direction, amount)

    @tool
    def go_back() -> str:
        """Go back to the previous page in browser history."""
        return session.go_back()

    @tool
    def finish(answer: str) -> str:
        """Call this ONCE when the task is complete (or impossible), with the
        final answer or result summary for the user. This ends the session."""
        return answer

    tools = [navigate, click, type_text, press_key, scroll, go_back, finish]
    if settings.agent_vision_mode == "hybrid":
        tools = [navigate, click_element, type_in_element, click, type_text,
                 press_key, scroll, go_back, finish]
    return tools
