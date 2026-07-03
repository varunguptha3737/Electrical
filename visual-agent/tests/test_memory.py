"""Memory vault tests: notes, recall, context injection, auto-extraction."""

from langchain_core.messages import AIMessage, HumanMessage

import visual_agent.memory as memory_mod
from visual_agent.chat import chat_system, stream_chat
from visual_agent.memory import MemoryVault, extract_and_store, get_vault


class FakeLLM:
    def __init__(self, invoke_reply="NONE", stream_tokens=("Hi", "!")):
        self.invoke_reply = invoke_reply
        self.stream_tokens = stream_tokens
        self.seen_messages = None

    def invoke(self, prompt):
        return AIMessage(content=self.invoke_reply)

    def stream(self, messages):
        self.seen_messages = messages
        for t in self.stream_tokens:
            yield AIMessage(content=t)


def _fresh_vault(tmp_path, monkeypatch) -> MemoryVault:
    monkeypatch.setattr("visual_agent.config.settings.memory_dir", str(tmp_path / "vault"))
    monkeypatch.setattr(memory_mod, "_vault", None)
    return get_vault()


def test_remember_recall_and_obsidian_format(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    p1 = vault.remember("The user's name is Varun.")
    vault.remember("The user prefers LibreOffice over Excel.")

    text = p1.read_text()
    assert text.startswith("---") and "tags:" in text  # Obsidian frontmatter

    hits = vault.recall("what is my name")
    assert hits and "Varun" in hits[0]

    ctx = vault.context("libreoffice")
    assert "LibreOffice" in ctx and ctx.startswith("- ")

    assert vault.forget(p1.name) is True
    assert vault.forget("nope.md") is False
    assert "Varun" not in " ".join(b for b in vault.recall("name"))


def test_extract_and_store(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    assert extract_and_store("hi", "hello!", FakeLLM("NONE")) is None
    assert len(vault.notes()) == 0
    path = extract_and_store(
        "my dog is called Bolt", "Nice name!", FakeLLM("The user's dog is called Bolt.")
    )
    assert path is not None and "Bolt" in path.read_text()


def test_stream_chat_injects_memory_and_logs(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    monkeypatch.setattr("visual_agent.config.settings.auto_memory", False)
    vault.remember("The user's name is Varun.")

    llm = FakeLLM(stream_tokens=("Hello ", "Varun!"))
    history = [HumanMessage(content="what is my name?")]
    reply = "".join(stream_chat(history, llm=llm))
    assert reply == "Hello Varun!"

    # memory was in the system prompt
    system = llm.seen_messages[0]
    assert "Varun" in system.content and "remember" in system.content.lower()

    # exchange was logged to today's transcript
    transcripts = list(vault.transcripts_dir.glob("*.md"))
    assert len(transcripts) == 1
    log = transcripts[0].read_text()
    assert "what is my name?" in log and "Hello Varun!" in log


def test_memory_disabled(tmp_path, monkeypatch):
    _fresh_vault(tmp_path, monkeypatch)
    monkeypatch.setattr("visual_agent.config.settings.memory_enabled", False)
    system = chat_system("anything")
    assert "remember" not in system.content.lower()
