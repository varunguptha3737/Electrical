"""Set-of-Marks grounding: extract the page's interactive elements and
overlay numbered boxes on the screenshot, so the model can act by element id
instead of guessing pixel coordinates."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

MAX_ELEMENTS = 80

_EXTRACT_JS = """
() => {
  const sel = 'a, button, input, select, textarea, summary, ' +
    '[role="button"], [role="link"], [role="tab"], [role="checkbox"], ' +
    '[role="radio"], [role="menuitem"], [role="combobox"], [role="option"], ' +
    '[role="switch"], [role="searchbox"], [onclick], [contenteditable="true"]';
  const seen = new Set();
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    if (seen.has(el)) continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    if (r.width < 3 || r.height < 3) continue;
    if (r.bottom < 0 || r.right < 0 ||
        r.top > window.innerHeight || r.left > window.innerWidth) continue;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' ||
        style.pointerEvents === 'none') continue;
    let name = el.getAttribute('aria-label') || el.innerText || el.value ||
      el.placeholder || el.getAttribute('title') || el.getAttribute('alt') || '';
    name = name.trim().replace(/\\s+/g, ' ').slice(0, 60);
    const role = el.getAttribute('role') ||
      (el.tagName === 'INPUT' ? 'input:' + (el.type || 'text')
                              : el.tagName.toLowerCase());
    out.push({role: role, name: name,
              x: Math.round(r.x), y: Math.round(r.y),
              width: Math.round(r.width), height: Math.round(r.height)});
  }
  return out;
}
"""


@dataclass
class PageElement:
    element_id: int
    role: str
    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def describe(self) -> str:
        name = f' "{self.name}"' if self.name else ""
        return f"[{self.element_id}] {self.role}{name}"


def extract_elements(page) -> list[PageElement]:
    """Enumerate visible interactive elements in the viewport, ids start at 1."""
    raw = page.evaluate(_EXTRACT_JS)[:MAX_ELEMENTS]
    return [PageElement(element_id=i + 1, **item) for i, item in enumerate(raw)]


def elements_summary(elements: list[PageElement]) -> str:
    if not elements:
        return "No interactive elements detected in the viewport."
    return "Interactive elements on screen:\n" + "\n".join(
        e.describe() for e in elements
    )


def annotate_screenshot(png: bytes, elements: list[PageElement]) -> bytes:
    """Draw numbered boxes (Set-of-Marks) on a screenshot PNG."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    for el in elements:
        box = (el.x, el.y, el.x + el.width, el.y + el.height)
        draw.rectangle(box, outline=(255, 60, 60), width=2)
        label = str(el.element_id)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        lx, ly = max(el.x - 1, 0), max(el.y - th - 4, 0)
        draw.rectangle((lx, ly, lx + tw + 6, ly + th + 4), fill=(255, 60, 60))
        draw.text((lx + 3, ly + 2), label, fill="white", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
