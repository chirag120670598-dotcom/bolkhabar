"""Word-level transcript model.

Compatible with the JSON that Whisper, faster-whisper and
[bolcut](https://github.com/chirag120670598-dotcom/bolcut) all produce, so the
same file can be edited as a video and repurposed as content:

```json
{"words": [{"word": "Namaste", "start": 0.0, "end": 0.4}]}
```

Timestamps are validated on load. A transcript with backwards or badly
overlapping words produces wrong chapter markers and wrong clip timings, and
that is much harder to notice than a crash.

Pure standard library.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "TranscriptError",
    "Word",
    "Sentence",
    "Transcript",
    "format_timestamp",
]

#: Two words may differ by this much and still be treated as adjacent —
#: automatic speech recognition rounds differently between engines.
OVERLAP_TOLERANCE = 0.001


class TranscriptError(ValueError):
    """Raised when a transcript is malformed."""


def format_timestamp(seconds: float, *, colon: bool = True) -> str:
    """Seconds → ``HH:MM:SS`` (or ``MM:SS`` under an hour).

    >>> format_timestamp(3725.4)
    '01:02:05'
    >>> format_timestamp(125)
    '02:05'
    """
    if seconds < 0:
        raise TranscriptError(f"timestamp cannot be negative: {seconds}")
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours or not colon:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class Word:
    """One word with its position on the timeline, in seconds."""

    text: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise TranscriptError(f"word {self.text!r} starts before 0")
        if self.end < self.start:
            raise TranscriptError(
                f"word {self.text!r} ends at {self.end} before it starts at "
                f"{self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def normalized(self) -> str:
        """Lowercase, punctuation stripped — what analysis compares on."""
        return re.sub(r"[^0-9a-z\u0900-\u097f]", "", self.text.lower())


@dataclass(frozen=True)
class Sentence:
    """A run of words ending in sentence-final punctuation (or a long pause)."""

    words: Tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def __str__(self) -> str:
        return self.text


@dataclass
class Transcript:
    """An ordered list of words covering one piece of media."""

    words: List[Word] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    title: str = ""

    def __post_init__(self) -> None:
        self.validate()

    # --- construction -----------------------------------------------------

    @classmethod
    def from_words(cls, words: Iterable[Sequence], *, language=None,
                   duration=None, title: str = "") -> "Transcript":
        return cls(
            words=[Word(str(t), float(s), float(e)) for t, s, e in words],
            language=language, duration=duration, title=title)

    @classmethod
    def from_dicts(cls, entries: Iterable[dict], **kw) -> "Transcript":
        """Whisper's shape: ``{"word": …, "start": …, "end": …}``."""
        out = []
        for entry in entries:
            text = entry.get("word") or entry.get("text") or ""
            out.append(Word(str(text).strip(),
                            float(entry.get("start", 0.0)),
                            float(entry.get("end", 0.0))))
        return cls(words=out, **kw)

    @classmethod
    def from_json(cls, raw: Union[str, bytes, dict, list], **kw) -> "Transcript":
        if isinstance(raw, (str, bytes)):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TranscriptError(f"transcript is not valid JSON: {exc}") from exc
        else:
            data = raw
        if isinstance(data, dict):
            entries = data.get("words") or data.get("segments") or []
            kw.setdefault("language", data.get("language"))
            kw.setdefault("duration", data.get("duration"))
            kw.setdefault("title", data.get("title", ""))
        elif isinstance(data, list):
            entries = data
        else:
            raise TranscriptError("transcript JSON must be an object or a list")
        if not entries:
            raise TranscriptError("transcript has no words")
        return cls.from_dicts(entries, **kw)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kw) -> "Transcript":
        path = Path(path)
        if not path.exists():
            raise TranscriptError(f"transcript file not found: {path}")
        return cls.from_json(path.read_text(encoding="utf-8"), **kw)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps({
            "language": self.language, "duration": self.duration,
            "title": self.title,
            "words": [{"word": w.text, "start": w.start, "end": w.end}
                      for w in self.words],
        }, ensure_ascii=False, indent=indent)

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """Reject transcripts that would silently produce wrong timings."""
        previous: Optional[Word] = None
        for word in self.words:
            if previous is not None:
                if word.start < previous.start - OVERLAP_TOLERANCE:
                    raise TranscriptError(
                        f"words out of order: {previous.text!r} at "
                        f"{previous.start} followed by {word.text!r} at "
                        f"{word.start}")
                if word.start < previous.end - OVERLAP_TOLERANCE:
                    raise TranscriptError(
                        f"words overlap: {previous.text!r} "
                        f"[{previous.start}, {previous.end}] and "
                        f"{word.text!r} [{word.start}, {word.end}]")
            previous = word

    # --- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words).strip()

    @property
    def total_duration(self) -> float:
        if self.duration is not None:
            return float(self.duration)
        return self.words[-1].end if self.words else 0.0

    @property
    def minutes(self) -> float:
        return self.total_duration / 60.0

    def sentences(self, *, max_pause: float = 0.8) -> List[Sentence]:
        """Group words into sentences.

        ASR transcripts have no punctuation, so a long pause between words is
        used as the boundary alongside ``. ! ?``.
        """
        if not self.words:
            return []
        current: List[Word] = []
        out: List[Sentence] = []
        for word in self.words:
            if current and word.start - current[-1].end > max_pause:
                out.append(Sentence(tuple(current)))
                current = []
            current.append(word)
            if word.text.rstrip().endswith((".", "!", "?", "\u0964")):
                out.append(Sentence(tuple(current)))
                current = []
        if current:
            out.append(Sentence(tuple(current)))
        return out

    def words_between(self, start: float, end: float) -> List[Word]:
        return [w for w in self.words if w.start >= start and w.end <= end]

    def text_between(self, start: float, end: float) -> str:
        return " ".join(w.text.strip() for w in self.words_between(start, end)).strip()
