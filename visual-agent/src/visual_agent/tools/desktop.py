"""OS-level computer use: control the whole desktop, not just a browser.

Opens applications (LibreOffice, GIMP, file manager, …) and acts on the
screen with mouse/keyboard via pyautogui, capturing screenshots with mss.
Works on Windows, macOS and Linux/X11 — the machine must have a display
(or a virtual one like Xvfb; see README).

The model sees screenshots downscaled to at most `max_image_edge` pixels;
`DesktopSession` remembers the scale factor and maps the model's click
coordinates back to real screen pixels.
"""

from __future__ import annotations

import io
import shlex
import subprocess
import time

from langchain_core.tools import tool
from PIL import Image

from ..config import settings

DESKTOP_SYSTEM_RULES = """\
- You are controlling the FULL desktop (any application), not just a browser.
- To click, give exact pixel coordinates of the center of the target as seen \
in the latest screenshot (x from left, y from top).
- Open applications with open_app using a shell command, e.g. \
open_app('libreoffice --calc'), open_app('gimp /home/user/photo.jpg'), \
open_app('nautilus'). Then wait for the window to appear before acting.
- Applications take time to start: use wait(2) after opening one, and check \
the next screenshot before clicking.
- Use hotkey for keyboard shortcuts, e.g. hotkey(['ctrl','s']) to save, \
hotkey(['ctrl','z']) to undo.
- Menus: click the menu name, wait for it to open in the next screenshot, \
then click the item."""


class DesktopSession:
    """Same surface as BrowserSession (screenshot/click/type/…) but for the
    whole screen. Input via pyautogui, capture via mss."""

    def __init__(self, max_image_edge: int | None = None):
        self._max_edge = max_image_edge or settings.max_image_edge
        self._pg = None
        self._sct = None
        self.native_width = 0
        self.native_height = 0
        self.scale = 1.0  # native pixels per screenshot pixel
        # width/height of the image the model sees (set on start)
        self.width = 0
        self.height = 0

    def start(self) -> "DesktopSession":
        import mss
        import pyautogui

        pyautogui.FAILSAFE = True  # slam mouse into a corner to abort
        self._pg = pyautogui
        self._sct = mss.mss()
        self.native_width, self.native_height = pyautogui.size()
        self.scale = max(
            1.0, max(self.native_width, self.native_height) / self._max_edge
        )
        self.width = round(self.native_width / self.scale)
        self.height = round(self.native_height / self.scale)
        return self

    def stop(self) -> None:
        if self._sct:
            self._sct.close()
        self._pg = self._sct = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _real(self, x: int, y: int) -> tuple[int, int]:
        return round(x * self.scale), round(y * self.scale)

    def screenshot(self) -> bytes:
        shot = self._sct.grab(self._sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if self.scale > 1.0:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Actions (x/y are screenshot coordinates) ------------------------------

    def open_app(self, command: str) -> str:
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2.5)
        return f"Launched {command!r}. Check the screenshot for its window."

    def click(self, x: int, y: int) -> str:
        self._pg.click(*self._real(x, y))
        time.sleep(0.4)
        return f"Clicked at ({x}, {y})."

    def double_click(self, x: int, y: int) -> str:
        self._pg.doubleClick(*self._real(x, y))
        time.sleep(0.4)
        return f"Double-clicked at ({x}, {y})."

    def right_click(self, x: int, y: int) -> str:
        self._pg.rightClick(*self._real(x, y))
        time.sleep(0.4)
        return f"Right-clicked at ({x}, {y})."

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> str:
        rx1, ry1 = self._real(x1, y1)
        rx2, ry2 = self._real(x2, y2)
        self._pg.moveTo(rx1, ry1)
        self._pg.dragTo(rx2, ry2, duration=0.6, button="left")
        return f"Dragged from ({x1}, {y1}) to ({x2}, {y2})."

    def type_text(self, text: str) -> str:
        self._pg.typewrite(text, interval=0.02)
        return f"Typed {text!r}."

    def press_key(self, key: str) -> str:
        self._pg.press(key.lower())
        time.sleep(0.3)
        return f"Pressed {key!r}."

    def hotkey(self, *keys: str) -> str:
        self._pg.hotkey(*[k.lower() for k in keys])
        time.sleep(0.4)
        return f"Pressed {'+'.join(keys)}."

    def scroll(self, direction: str, amount: int = 500) -> str:
        self._pg.scroll(-amount if direction == "down" else amount)
        time.sleep(0.3)
        return f"Scrolled {direction} by {amount}."


def make_desktop_tools(session: DesktopSession) -> list:
    @tool
    def open_app(command: str) -> str:
        """Launch an application with a shell command, e.g.
        'libreoffice --calc', 'gimp /path/to/photo.jpg', 'firefox'."""
        return session.open_app(command)

    @tool
    def click(x: int, y: int) -> str:
        """Click at pixel coordinates (x, y) on the latest screenshot."""
        return session.click(x, y)

    @tool
    def double_click(x: int, y: int) -> str:
        """Double-click at (x, y) — open files/folders, select words."""
        return session.double_click(x, y)

    @tool
    def right_click(x: int, y: int) -> str:
        """Right-click at (x, y) to open a context menu."""
        return session.right_click(x, y)

    @tool
    def drag(x1: int, y1: int, x2: int, y2: int) -> str:
        """Drag with the left button from (x1, y1) to (x2, y2) — move things,
        select regions, adjust sliders."""
        return session.drag(x1, y1, x2, y2)

    @tool
    def type_text(text: str) -> str:
        """Type text with the keyboard into whatever is focused."""
        return session.type_text(text)

    @tool
    def press_key(key: str) -> str:
        """Press one key: 'enter', 'tab', 'escape', 'delete', 'down', 'f2'…"""
        return session.press_key(key)

    @tool
    def hotkey(keys: list[str]) -> str:
        """Press a key combination, e.g. ['ctrl','s'] or ['alt','tab']."""
        return session.hotkey(*keys)

    @tool
    def scroll(direction: str, amount: int = 500) -> str:
        """Scroll 'up' or 'down' under the mouse cursor."""
        return session.scroll(direction, amount)

    @tool
    def wait(seconds: float) -> str:
        """Wait for the screen to change (app starting, file loading)."""
        time.sleep(min(seconds, 10))
        return f"Waited {seconds}s."

    @tool
    def finish(answer: str) -> str:
        """Call ONCE when the task is complete (or impossible), with the final
        result summary for the user. This ends the session."""
        return answer

    return [
        open_app, click, double_click, right_click, drag, type_text,
        press_key, hotkey, scroll, wait, finish,
    ]
