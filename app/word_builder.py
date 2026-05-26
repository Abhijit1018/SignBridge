import time


class WordBuilder:
    def __init__(self, stable_frames=20, space_gap=1.5, sentence_gap=3.0):
        self.stable_frames = stable_frames
        self.space_gap = space_gap
        self.sentence_gap = sentence_gap
        self.word_buffer: list[str] = []
        self._stable_label: str | None = None
        self._stable_count = 0
        self._last_confident_time = time.monotonic()
        self._last_appended: str | None = None

    def process(self, label: str | None, confident: bool) -> list[dict]:
        """
        Call once per frame. Returns list of events:
          {"type": "letter_added", "letter": X}
          {"type": "space_added"}
          {"type": "sentence_ready", "sentence": "..."}
        """
        events: list[dict] = []
        now = time.monotonic()

        if confident and label:
            self._last_confident_time = now
            if label == self._stable_label:
                self._stable_count += 1
            else:
                self._stable_label = label
                self._stable_count = 1
                self._last_appended = None

            if self._stable_count >= self.stable_frames and label != self._last_appended:
                self.word_buffer.append(label)
                self._last_appended = label
                self._stable_count = 0
                events.append({"type": "letter_added", "letter": label})
        else:
            self._stable_label = None
            self._stable_count = 0
            elapsed = now - self._last_confident_time

            if (elapsed > self.space_gap
                    and self.word_buffer
                    and self.word_buffer[-1] != " "):
                self.word_buffer.append(" ")
                events.append({"type": "space_added"})

            stripped = "".join(self.word_buffer).strip()
            if elapsed > self.sentence_gap and stripped:
                events.append({"type": "sentence_ready", "sentence": stripped})
                self.word_buffer.clear()
                self._last_confident_time = now  # prevent re-firing

        return events

    def clear(self):
        self.word_buffer.clear()
        self._stable_label = None
        self._stable_count = 0
        self._last_appended = None

    def delete_last(self):
        if self.word_buffer:
            self.word_buffer.pop()

    @property
    def current_word(self) -> str:
        return "".join(self.word_buffer)

    @property
    def stable_progress(self) -> float:
        if self._stable_count == 0:
            return 0.0
        return min(self._stable_count / self.stable_frames, 1.0)
