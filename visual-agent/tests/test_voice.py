"""Voice-pipeline unit tests (no audio models loaded)."""

from langchain_core.messages import AIMessage

from visual_agent.voice import narrate_action, route_intent, sentence_chunks


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.reply)


def test_sentence_chunks_splits_at_sentence_ends():
    tokens = ["Hello there", ". ", "How are ", "you today? ", "Fin"]
    chunks = list(sentence_chunks(tokens, min_len=5))
    assert chunks == ["Hello there.", "How are you today?", "Fin"]


def test_sentence_chunks_respects_min_len():
    # "Hi." is shorter than min_len → merged with the next sentence
    chunks = list(sentence_chunks(["Hi. ", "Nice to meet you."], min_len=10))
    assert chunks == ["Hi. Nice to meet you."]


def test_route_intent():
    assert route_intent("open google and search cats", FakeLLM("browse")) == "browse"
    assert route_intent("what do you see in this image", FakeLLM("chat")) == "chat"
    assert route_intent("hello", FakeLLM("CHAT.")) == "chat"


def test_narrate_action():
    assert "wikipedia" in narrate_action("navigate", {"url": "wikipedia.org"})
    assert "Launching" in narrate_action("open_app", {"command": "gimp"})
    assert narrate_action("press_key", {"key": "Enter"}) is None
