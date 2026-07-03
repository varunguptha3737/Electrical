"""Long-term memory as a folder of plain markdown notes.

The vault is Obsidian-compatible: open MEMORY_DIR as an Obsidian vault to
browse, edit or delete what the agent remembers. Layout:

    memory/
      notes/        one small .md note per remembered fact (YAML frontmatter)
      transcripts/  daily conversation logs

Recall is simple keyword scoring over the notes — transparent and fast, no
database or embedding model needed.
"""

from __future__ import annotations

import datetime as _dt
import re
import threading
from pathlib import Path

from .config import settings

EXTRACT_PROMPT = """You maintain long-term memory for a personal assistant.
Review this exchange. If it reveals a durable fact worth remembering about \
the user for FUTURE conversations (their name, preferences, projects, \
people, decisions, recurring tasks), state that fact in ONE short sentence.
If nothing is worth remembering long-term (small talk, one-off questions), \
reply with exactly: NONE

User: {user}
Assistant: {assistant}

Memory:"""

_WORD = re.compile(r"[a-z0-9]{3,}")


def _now() -> _dt.datetime:
    return _dt.datetime.now()


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "note"


class MemoryVault:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or settings.memory_dir)
        self.notes_dir = self.root / "notes"
        self.transcripts_dir = self.root / "transcripts"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    # -- writing --------------------------------------------------------

    def remember(self, fact: str, topic: str = "memory") -> Path:
        """Store one fact as a small markdown note."""
        fact = fact.strip()
        now = _now()
        path = self.notes_dir / f"{now:%Y%m%d-%H%M%S}-{_slug(fact)}.md"
        path.write_text(
            f"---\ncreated: {now:%Y-%m-%d %H:%M}\ntags: [{topic}]\n---\n\n{fact}\n",
            encoding="utf-8",
        )
        return path

    def forget(self, filename: str) -> bool:
        path = self.notes_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def log_exchange(self, user_text: str, assistant_text: str, mode: str = "chat") -> None:
        """Append the exchange to today's transcript."""
        now = _now()
        path = self.transcripts_dir / f"{now:%Y-%m-%d}.md"
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n## {now:%H:%M} ({mode})\n\n"
                f"**User:** {user_text.strip()}\n\n"
                f"**Assistant:** {assistant_text.strip()}\n"
            )

    # -- reading --------------------------------------------------------

    def notes(self) -> list[tuple[Path, str]]:
        """All notes, oldest first."""
        return [
            (p, p.read_text(encoding="utf-8"))
            for p in sorted(self.notes_dir.glob("*.md"))
        ]

    @staticmethod
    def _body(text: str) -> str:
        """Strip YAML frontmatter."""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                return parts[2].strip()
        return text.strip()

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Notes most relevant to the query (keyword scoring), best first."""
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for path, text in self.notes():
            body = self._body(text)
            words = _WORD.findall(body.lower())
            score = sum(words.count(t) for t in terms)
            if score > 0:
                scored.append((score, path.stat().st_mtime, body))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        return [body for _, _, body in scored[:limit]]

    def context(self, query: str = "", max_chars: int = 1500) -> str:
        """Memory block for the system prompt: relevant notes first, then the
        most recent ones, deduplicated and capped."""
        picked: list[str] = []
        for body in self.recall(query) + [
            self._body(t) for _p, t in reversed(self.notes())
        ]:
            if body not in picked:
                picked.append(body)
        out, used = [], 0
        for body in picked:
            if used + len(body) > max_chars:
                break
            out.append(f"- {body}")
            used += len(body)
        return "\n".join(out)


_vault: MemoryVault | None = None


def get_vault() -> MemoryVault:
    global _vault
    if _vault is None or _vault.root != Path(settings.memory_dir):
        _vault = MemoryVault()
    return _vault


# -- automatic memory extraction ------------------------------------------


def extract_and_store(user_text: str, assistant_text: str, llm) -> Path | None:
    """One LLM call: did this exchange reveal a durable fact? Store it if so."""
    reply = llm.invoke(
        EXTRACT_PROMPT.format(user=user_text[:2000], assistant=assistant_text[:2000])
    )
    fact = (reply.content if hasattr(reply, "content") else str(reply)).strip()
    if not fact or fact.upper().startswith("NONE"):
        return None
    return get_vault().remember(fact)


def after_exchange(user_text: str, assistant_text: str, llm, mode: str = "chat") -> None:
    """Post-exchange bookkeeping: log the transcript and (in a background
    thread, so replies aren't delayed) auto-extract memories."""
    if not settings.memory_enabled or not assistant_text.strip():
        return
    vault = get_vault()
    vault.log_exchange(user_text, assistant_text, mode)
    if settings.auto_memory:
        threading.Thread(
            target=lambda: extract_and_store(user_text, assistant_text, llm),
            daemon=True,
        ).start()
