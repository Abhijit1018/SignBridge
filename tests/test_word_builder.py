import time
import pytest
from app.word_builder import WordBuilder


def wb():
    """WordBuilder with short timeouts for fast tests."""
    return WordBuilder(stable_frames=3, space_gap=0.05, sentence_gap=0.1)


def test_letter_added_after_stable_frames():
    b = wb()
    events = []
    for _ in range(3):
        events.extend(b.process("A", confident=True))
    types = [e["type"] for e in events]
    assert "letter_added" in types
    assert "A" in b.word_buffer


def test_same_letter_not_added_twice_in_a_row():
    b = wb()
    for _ in range(9):
        b.process("A", confident=True)
    assert b.word_buffer.count("A") == 1


def test_different_letter_resets_stable_count():
    b = wb()
    for _ in range(2):
        b.process("A", confident=True)
    for _ in range(3):
        b.process("B", confident=True)
    assert "A" not in b.word_buffer
    assert "B" in b.word_buffer


def test_space_added_after_space_gap():
    b = wb()
    for _ in range(3):
        b.process("A", confident=True)
    assert b.word_buffer == ["A"]
    time.sleep(0.06)
    b.process(None, confident=False)
    assert " " in b.word_buffer


def test_space_not_added_twice():
    b = wb()
    for _ in range(3):
        b.process("A", confident=True)
    time.sleep(0.06)
    b.process(None, confident=False)
    b.process(None, confident=False)
    assert b.word_buffer.count(" ") == 1


def test_sentence_ready_after_sentence_gap():
    b = wb()
    for _ in range(3):
        b.process("A", confident=True)
    time.sleep(0.12)
    events = b.process(None, confident=False)
    ready = [e for e in events if e["type"] == "sentence_ready"]
    assert len(ready) == 1
    assert ready[0]["sentence"] == "A"
    assert b.word_buffer == []


def test_sentence_not_fired_when_buffer_empty():
    b = wb()
    time.sleep(0.12)
    events = b.process(None, confident=False)
    ready = [e for e in events if e["type"] == "sentence_ready"]
    assert len(ready) == 0


def test_clear_resets_buffer():
    b = wb()
    for _ in range(3):
        b.process("A", confident=True)
    b.clear()
    assert b.word_buffer == []


def test_delete_last():
    b = wb()
    for _ in range(3):
        b.process("A", confident=True)
    b.delete_last()
    assert "A" not in b.word_buffer


def test_stable_progress_increases():
    b = wb()
    b.process("A", confident=True)
    b.process("A", confident=True)
    assert b.stable_progress == pytest.approx(2 / 3)


def test_stable_progress_resets_on_different_letter():
    b = wb()
    b.process("A", confident=True)
    b.process("B", confident=True)
    assert b.stable_progress == pytest.approx(1 / 3)


def test_current_word_property():
    b = wb()
    for _ in range(3):
        b.process("H", confident=True)
    for _ in range(3):
        b.process("I", confident=True)
    assert b.current_word == "HI"
