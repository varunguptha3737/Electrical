"""Desktop-session unit tests with a mocked input backend (no display needed)."""

import io

from PIL import Image

from visual_agent.tools.desktop import DesktopSession, make_desktop_tools


class FakePG:
    """Records pyautogui calls."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record


def make_session(native=(2560, 1440), max_edge=1280) -> tuple[DesktopSession, FakePG]:
    s = DesktopSession(max_image_edge=max_edge)
    pg = FakePG()
    s._pg = pg
    s.native_width, s.native_height = native
    s.scale = max(1.0, max(native) / max_edge)
    s.width = round(native[0] / s.scale)
    s.height = round(native[1] / s.scale)
    return s, pg


def test_coordinates_scale_back_to_native_pixels():
    s, pg = make_session(native=(2560, 1440), max_edge=1280)  # scale 2.0
    assert (s.width, s.height) == (1280, 720)
    s.click(100, 50)
    assert pg.calls[0] == ("click", (200, 100))
    s.drag(0, 0, 640, 360)
    assert ("dragTo", (1280, 720)) in [(n, a[:2]) for n, a in pg.calls if n == "dragTo"]


def test_no_upscale_on_small_screens():
    s, pg = make_session(native=(1024, 768), max_edge=1280)  # scale stays 1.0
    s.click(300, 200)
    assert pg.calls[0] == ("click", (300, 200))


def test_hotkey_and_keys_lowercased():
    s, pg = make_session()
    s.hotkey("Ctrl", "S")
    assert pg.calls[0] == ("hotkey", ("ctrl", "s"))
    s.press_key("Enter")
    assert pg.calls[1] == ("press", ("enter",))


def test_desktop_toolset():
    s, _pg = make_session()
    names = {t.name for t in make_desktop_tools(s)}
    assert {"open_app", "click", "double_click", "drag", "hotkey", "wait",
            "finish"} <= names
