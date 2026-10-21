"""Content analysis: keywords, chapters, quotes and summaries.

No model, no embeddings, no API key. Every technique here is a
frequency-and-position heuristic, which means:

* it runs offline, in milliseconds, for free
* the same episode always produces the same chapters
* you can read the code and see exactly why a segment was chosen

That is a real trade-off. A transformer would produce better topic labels.
But a repurposer that costs money and 30 seconds per episode, and gives a
different answer each run, is worse for a creator publishing on a schedule.

.. note::
   The Hindi stopword list is small on purpose. Over-filtering Hinglish is
   worse than under-filtering: a word wrongly removed disappears from every
   chapter title, keyword list and quote score.

Pure standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .transcript import Sentence, Transcript, format_timestamp

__all__ = [
    "ENGLISH_STOPWORDS",
    "HINDI_STOPWORDS",
    "FILLERS",
    "Keyword",
    "Chapter",
    "Quote",
    "top_keywords",
    "segment_chapters",
    "extract_quotes",
    "summarize",
    "tokenize",
]

ENGLISH_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could didn't do does
doing don't down during each few for from further had has have having he her
here hers herself him himself his how i if in into is isn't it its itself just
me more most my myself no nor not now of off on once only or other our ours
out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with you your yours
""".split())

#: Hindi/Hinglish function words, in roman spelling.
HINDI_STOPWORDS = frozenset("""
ka ke ki ko se me mein hai hain tha thi the aur ya to bhi hi toh par lekin magar
ek do teen is us ye vo yeh voh jo ki wala wali wale ne ko tak sirf bahut thoda
zyada kam sab kuch koi apna mera tera humara aap aapka main maine tum tumhe
""".split())

FILLERS = frozenset("""
um uh er like literally actually basically basically matlab yani accha achha
arre arey yaar see you know mean thing things stuff kinda sorta gonna wanna
""".split())

_WORD_RE = re.compile(r"[a-z\u0900-\u097f]{2,}")


def tokenize(text: str) -> List[str]:
    """Lowercase words, stopwords and fillers removed."""
    out = []
    for match in _WORD_RE.findall(text.lower()):
        if match in ENGLISH_STOPWORDS or match in HINDI_STOPWORDS:
            continue
        if match in FILLERS:
            continue
        out.append(match)
    return out


@dataclass(frozen=True)
class Keyword:
    term: str
    count: int
    score: float

    def __str__(self) -> str:
        return self.term


def top_keywords(text: str, *, n: int = 10,
                 bigrams: bool = True) -> List[Keyword]:
    """Most significant terms, by TF scaled by how spread out they are.

    A word repeated in one burst is a tangent. A word spread across the
    episode is the topic. So each term's count is divided by the number of
    distinct positions it appears in, raised to a power below 1 — rewarding
    frequency while damping single-burst repetition.
    """
    words = tokenize(text)
    if not words:
        return []

    counts = Counter(words)
    positions: Dict[str, set] = defaultdict(set)
    for index, word in enumerate(words):
        positions[word].add(index // 50)   # 50-word buckets

    scored = []
    for term, count in counts.items():
        spread = len(positions[term])
        score = count / (1 + math.log(1 + spread))
        scored.append(Keyword(term, count, score))
    scored.sort(key=lambda k: -k.score)
    results = [k for k in scored if k.count >= 2][:n]

    if bigrams:
        pairs = Counter(zip(words, words[1:]))
        for (a, b), count in pairs.most_common(n * 3):
            if count < 3 or a == b:
                continue
            # a bigram that is just two keywords already listed adds nothing
            if any(k.term == f"{a} {b}" for k in results):
                continue
            results.append(Keyword(f"{a} {b}", count, count * 1.4))
            if len(results) >= n + 3:
                break

    return sorted(results, key=lambda k: -k.score)[:n]


@dataclass
class Chapter:
    """One topic segment of the episode."""

    start: float
    end: float
    title: str
    keywords: List[Keyword] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences).strip()

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end,
            "timestamp": self.timestamp, "title": self.title,
            "duration": round(self.duration, 2),
            "keywords": [k.term for k in self.keywords],
        }


def _title_from_keywords(keywords: Sequence[Keyword], limit: int = 6) -> str:
    """A chapter title from its strongest keywords, title-cased."""
    if not keywords:
        return "Untitled segment"
    words = []
    for keyword in keywords[:limit]:
        for part in keyword.term.split():
            if part not in words:
                words.append(part)
        if len(words) >= limit:
            break
    return " ".join(w.capitalize() for w in words[:limit])


def segment_chapters(
    transcript: Transcript,
    *,
    target_minutes: float = 5.0,
    min_chapters: int = 3,
    max_chapters: int = 20,
) -> List[Chapter]:
    """Split the episode into topic segments.

    Boundaries are placed at sentence gaps where the surrounding vocabulary
    changes the most — the same signal a listener uses to notice "ok, new
    topic". That beats fixed-length chunks, which cut mid-sentence, and beats
    pure keyword clustering, which needs a model.
    """
    sentences = transcript.sentences()
    if not sentences:
        return []
    if len(sentences) <= 1:
        return [Chapter(sentences[0].start, sentences[0].end,
                        _title_from_keywords(top_keywords(sentences[0].text, n=4)),
                        [], [sentences[0]])]

    total = transcript.total_duration
    target_chapters = max(
        min_chapters,
        min(max_chapters, int(round(total / (target_minutes * 60))) or min_chapters))
    target_chapters = min(target_chapters, len(sentences))

    # vocabulary shift between consecutive sentences
    vectors = [Counter(tokenize(s.text)) for s in sentences]
    shifts: List[float] = []
    for i in range(len(vectors) - 1):
        left = sum(vectors[i].values())
        right = sum(vectors[i + 1].values())
        if not left or not right:
            shifts.append(0.0)
            continue
        overlap = sum((vectors[i] & vectors[i + 1]).values())
        shifts.append(1.0 - (2 * overlap) / (left + right))

    # a boundary also needs a pause — no mid-thought cuts
    pauses = [sentences[i + 1].start - sentences[i].end
              for i in range(len(sentences) - 1)]
    max_pause = max(pauses) if pauses else 1.0
    max_shift = max(shifts) if shifts else 1.0

    scores = [
        (shifts[i] / max_shift if max_shift else 0) * 0.7
        + (pauses[i] / max_pause if max_pause else 0) * 0.3
        for i in range(len(shifts))
    ]

    # enforce a minimum spacing so chapters are not clustered
    min_gap = max(1, len(sentences) // (target_chapters * 2))
    chosen: List[int] = []
    for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if all(abs(index - c) >= min_gap for c in chosen):
            chosen.append(index)
        if len(chosen) >= target_chapters - 1:
            break
    boundaries = sorted(chosen)

    chapters: List[Chapter] = []
    start_index = 0
    for boundary in (*boundaries, len(sentences) - 1):
        end_index = boundary + 1 if boundary < len(sentences) - 1 else len(sentences)
        chunk = sentences[start_index:end_index]
        if not chunk:
            continue
        text = " ".join(s.text for s in chunk)
        keywords = top_keywords(text, n=5)
        chapters.append(Chapter(
            start=chunk[0].start, end=chunk[-1].end,
            title=_title_from_keywords(keywords), keywords=keywords,
            sentences=chunk))
        start_index = end_index
    return chapters


@dataclass(frozen=True)
class Quote:
    """A pull-out quote, with where it lives on the timeline."""

    text: str
    start: float
    end: float
    score: float
    reasons: Tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "start": self.start, "end": self.end,
            "timestamp": self.timestamp, "score": round(self.score, 3),
            "duration": round(self.duration, 2), "reasons": list(self.reasons),
        }


_STRONG = frozenset("""
never always everyone nobody secret truth mistake reality actually problem
solution important biggest worst best first time remember forget
""".split())

_HINDI_STRONG = frozenset("""
sabse zaroori galat sahi sach baat problem solution yaad rakho dhyan dena
matlab simple seedhi
""".split())


def extract_quotes(
    transcript: Transcript,
    *,
    n: int = 5,
    min_words: int = 8,
    max_words: int = 45,
    include_questions: bool = False,
) -> List[Quote]:
    """Find the sentences worth putting on a thumbnail or a card.

    Scoring is deliberately transparent — every quote carries the reasons it
    was picked, so you can see why and disagree.
    """
    sentences = transcript.sentences()
    if not sentences:
        return []

    all_words = tokenize(transcript.text)
    keyword_terms = {k.term for k in top_keywords(transcript.text, n=25)}

    scored: List[Quote] = []
    for sentence in sentences:
        if not (min_words <= sentence.word_count <= max_words):
            continue

        text = sentence.text
        reasons: List[str] = []
        score = 0.0

        # 1. it contains the episode's own keywords
        hits = [w for w in tokenize(text) if w in keyword_terms]
        if hits:
            score += min(2.0, len(hits) * 0.4)
            reasons.append(f"{len(hits)} keyword(s)")

        # 2. it makes a strong claim
        lowered = text.lower()
        strong = [w for w in _STRONG if f" {w} " in f" {lowered} "]
        hindi = [w for w in _HINDI_STRONG if w in lowered]
        if strong or hindi:
            score += 0.8
            reasons.append("strong claim")

        # 3. numbers are quotable — people repeat them
        if re.search(r"\b\d+\b", text):
            score += 0.6
            reasons.append("contains a number")

        # 4. a clean sentence ending reads better as a pull quote
        if text.rstrip().endswith((".", "!", "?", "\u0964")):
            score += 0.3
            reasons.append("clean ending")

        # 5. a question makes a bad pull quote — there is nothing on the card
        #    to read. Excluded by default rather than merely penalised, because
        #    a penalty can still leave a question above the cut.
        if text.rstrip().endswith("?"):
            if not include_questions:
                continue
            score -= 0.7
            reasons.append("question (penalised)")

        # 6. penalise filler-heavy speech
        fillers = sum(1 for w in tokenize(text) if w in FILLERS)
        if fillers:
            score -= fillers * 0.3

        # 7. mid-episode sentences are usually the substance; intros are not
        position = sentence.start / max(transcript.total_duration, 1.0)
        if 0.05 < position < 0.95:
            score += 0.2

        if score > 0.8:
            scored.append(Quote(text, sentence.start, sentence.end,
                                round(score, 3), tuple(reasons)))

    scored.sort(key=lambda q: -q.score)

    # drop near-duplicates: the same point said twice in a row
    out: List[Quote] = []
    for quote in scored:
        if any(_similar(quote.text, kept.text) for kept in out):
            continue
        out.append(quote)
        if len(out) >= n:
            break
    return out


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Cheap set-overlap similarity — good enough to spot a repeated point."""
    words_a, words_b = set(tokenize(a)), set(tokenize(b))
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b)) >= threshold


def summarize(
    transcript: Transcript,
    *,
    max_sentences: int = 5,
    prefer_start: bool = True,
) -> List[Sentence]:
    """Extractive summary: the highest-value sentences, in original order.

    Extractive rather than generative, so it can never invent something the
    guest did not say. For a podcast that matters — a hallucinated summary
    quote gets screenshotted.
    """
    sentences = transcript.sentences()
    if not sentences:
        return []

    keyword_terms = {k.term for k in top_keywords(transcript.text, n=30)}
    total = max(transcript.total_duration, 1.0)

    scored: List[Tuple[float, int, Sentence]] = []
    for index, sentence in enumerate(sentences):
        if sentence.word_count < 5:
            continue
        score = 0.0
        hits = sum(1 for w in tokenize(sentence.text) if w in keyword_terms)
        score += min(3.0, hits * 0.5)
        position = sentence.start / total
        if prefer_start and position < 0.15:
            score += 0.8          # the thesis usually lands early
        if 0.05 < position < 0.9:
            score += 0.2
        if sentence.word_count > 40:
            score -= 0.3          # rambling
        scored.append((score, index, sentence))

    scored.sort(key=lambda t: -t[0])
    picked = scored[:max_sentences]
    if prefer_start:
        picked.sort(key=lambda t: t[1])
    return [sentence for _, _, sentence in picked]
