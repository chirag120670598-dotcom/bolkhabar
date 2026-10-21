"""Content writers: thread, blog, newsletter, show notes, shorts, SEO.

Each writer turns the same analysis into a different shape. They are
deterministic and offline — no LLM call. If you want LLM polish, plug it in
at the adapter layer; the structure below is what makes the output usable
either way.

Everything respects real platform constraints:

* **X/Twitter** — 280 characters per tweet, split on sentence boundaries,
  never mid-word, with a ``1/N`` counter and a hook on the first tweet
* **blog** — H2 per chapter, pull quotes interleaved, word count estimate
* **shorts** — 30–60 second vertical candidates with the actual clip range
* **SEO** — title under 60 chars, meta description under 155

Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .analysis import Chapter, Keyword, Quote, top_keywords
from .transcript import Sentence, Transcript, format_timestamp

__all__ = [
    "TWEET_LIMIT",
    "TITLE_LIMIT",
    "META_LIMIT",
    "split_thread",
    "build_thread",
    "build_blog",
    "build_newsletter",
    "build_show_notes",
    "build_shorts",
    "build_seo",
    "Short",
    "SEO",
]

TWEET_LIMIT = 280
TITLE_LIMIT = 60
META_LIMIT = 155

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


# --- threads ----------------------------------------------------------------

def split_thread(text: str, *, limit: int = TWEET_LIMIT,
                 prefix: str = "") -> List[str]:
    """Split prose into tweets that fit the character limit.

    Rules, in order:

    1. split on sentence boundaries
    2. if a single sentence is still too long, split on commas
    3. if that fails, split on spaces — never mid-word
    4. reserve room for the ``1/N`` counter on every tweet

    >>> split_thread("Short one.", limit=280)
    ['Short one.']
    """
    budget = limit - len(prefix) - 4      # leave room for " 12/34"
    if budget <= 0:
        raise ValueError(f"limit {limit} is too small for a thread prefix")

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    chunks: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in sentences:
        if len(sentence) > budget:
            # too long on its own: break it down further
            flush()
            pieces = []
            for part in sentence.split(", "):
                if len(part) > budget:
                    words = part.split()
                    buf = ""
                    for word in words:
                        if len(buf) + len(word) + 1 > budget:
                            pieces.append(buf.strip())
                            buf = word
                        else:
                            buf = f"{buf} {word}".strip()
                    if buf:
                        pieces.append(buf)
                else:
                    pieces.append(part)
            for piece in pieces:
                if len(current) + len(piece) + 2 <= budget:
                    current = f"{current} {piece}".strip()
                else:
                    flush()
                    current = piece
            continue

        if len(current) + len(sentence) + 1 <= budget:
            current = f"{current} {sentence}".strip()
        else:
            flush()
            current = sentence
    flush()
    return chunks


def build_thread(
    transcript: Transcript,
    *,
    chapters: Sequence[Chapter] = (),
    quotes: Sequence[Quote] = (),
    hook: str = "",
    cta: str = "",
    limit: int = TWEET_LIMIT,
    max_tweets: int = 15,
) -> List[str]:
    """An X thread: hook, one tweet per chapter, pull quotes, CTA.

    Every tweet is guaranteed to be within ``limit`` including its counter,
    because that is the thing that actually breaks when you paste it.
    """
    chapters = list(chapters) or [
        Chapter(0.0, transcript.total_duration, transcript.title or "Episode",
                top_keywords(transcript.text, n=4), transcript.sentences())]

    body: List[str] = []
    if hook:
        body.append(hook)
    for index, chapter in enumerate(chapters, start=1):
        text = f"{index}. {chapter.title} [{chapter.timestamp}]"
        quote = next((q for q in quotes
                      if chapter.start <= q.start <= chapter.end), None)
        if quote:
            text += f"\n\n\u201c{quote.text}\u201d"
        body.append(text)
    if cta:
        body.append(cta)

    # Reserve room for the widest possible counter before splitting, not a
    # fixed guess: "  10/15" is 7 characters, so a fixed 6 overflows the limit
    # the moment a thread goes past 9 tweets.
    suffix = len(f"  {max_tweets}/{max_tweets}")
    budget = max(20, limit - suffix)

    expanded: List[str] = []
    for item in body:
        if len(item) <= budget:
            expanded.append(item)
        else:
            expanded.extend(split_thread(item, limit=budget))

    expanded = expanded[:max_tweets]
    count = len(expanded)
    return [f"{text}  {i}/{count}" for i, text in enumerate(expanded, start=1)]


# --- blog -------------------------------------------------------------------

@dataclass
class BlogPost:
    title: str
    description: str
    body: str
    word_count: int = 0
    read_minutes: int = 0

    def to_dict(self) -> dict:
        return {"title": self.title, "description": self.description,
                "body": self.body, "word_count": self.word_count,
                "read_minutes": self.read_minutes}


def build_blog(
    transcript: Transcript,
    *,
    chapters: Sequence[Chapter] = (),
    quotes: Sequence[Quote] = (),
    summary: Sequence[Sentence] = (),
    title: str = "",
    intro: str = "",
    outro: str = "",
    words_per_minute: int = 220,
) -> BlogPost:
    """A blog post with one H2 per chapter and pull quotes interleaved.

    The body is assembled from the *actual transcript text*, not paraphrased,
    so nothing is invented. Clean it up by hand or pass it through an LLM
    adapter — but the facts come from what was said.
    """
    chapters = list(chapters)
    heading = title or transcript.title or "Episode notes"
    keywords = top_keywords(transcript.text, n=8)

    description = " ".join(s.text for s in summary[:2])[:META_LIMIT]
    if not description:
        description = (f"{', '.join(k.term for k in keywords[:5])}. "
                       f"{format_timestamp(transcript.total_duration)} episode.")

    parts: List[str] = [f"# {heading}", ""]
    if intro:
        parts += [intro, ""]
    if summary:
        parts += ["## In short", ""]
        parts += [f"- {sentence.text}" for sentence in summary]
        parts.append("")

    for index, chapter in enumerate(chapters, start=1):
        parts += [f"## {index}. {chapter.title}", ""]
        parts += [f"*[{chapter.timestamp}]*", ""]
        for sentence in chapter.sentences[:6]:
            parts += [sentence.text, ""]
        quote = next((q for q in quotes
                      if chapter.start <= q.start <= chapter.end), None)
        if quote:
            parts += [f"> \u201c{quote.text}\u201d", ""]

    if keywords:
        parts += ["## Keywords", "",
                  ", ".join(k.term for k in keywords), ""]
    if outro:
        parts += [outro, ""]

    body = "\n".join(parts).strip() + "\n"
    words = len(re.findall(r"\S+", body))
    return BlogPost(
        title=heading, description=description, body=body,
        word_count=words,
        read_minutes=max(1, round(words / words_per_minute)))


# --- newsletter -------------------------------------------------------------

def build_newsletter(
    transcript: Transcript,
    *,
    chapters: Sequence[Chapter] = (),
    quotes: Sequence[Quote] = (),
    summary: Sequence[Sentence] = (),
    title: str = "",
    subject: str = "",
    listen_url: str = "",
) -> dict:
    """A newsletter payload: subject line, preview text, sections."""
    keywords = top_keywords(transcript.text, n=6)
    subject = subject or title or transcript.title or "New episode"
    if len(subject) > 60:
        subject = subject[:57].rstrip() + "..."

    preview = " ".join(s.text for s in summary[:1])[:110]
    if not preview and keywords:
        preview = "This episode: " + ", ".join(k.term for k in keywords[:4])

    sections = [
        {"heading": chapter.title, "timestamp": chapter.timestamp,
         "text": " ".join(s.text for s in chapter.sentences[:4]),
         "keywords": [k.term for k in chapter.keywords[:3]]}
        for chapter in chapters
    ]
    return {
        "subject": subject,
        "preview": preview,
        "duration": format_timestamp(transcript.total_duration),
        "sections": sections,
        "quotes": [q.to_dict() for q in quotes[:3]],
        "takeaways": [s.text for s in summary[:5]],
        "listen_url": listen_url,
        "keywords": [k.term for k in keywords],
    }


# --- show notes -------------------------------------------------------------

def build_show_notes(
    transcript: Transcript,
    *,
    chapters: Sequence[Chapter] = (),
    quotes: Sequence[Quote] = (),
    summary: Sequence[Sentence] = (),
    title: str = "",
    guests: Sequence[str] = (),
) -> str:
    """Timestamped show notes, ready to paste into a description box."""
    keywords = top_keywords(transcript.text, n=12)
    out: List[str] = []
    heading = title or transcript.title or "Episode"
    out += [heading, "=" * len(heading), ""]
    out.append(f"Duration: {format_timestamp(transcript.total_duration)}")
    if guests:
        out.append(f"Guests: {', '.join(guests)}")
    out.append("")

    if summary:
        out += ["IN SHORT", "-"]
        out += [f"\u2022 {s.text}" for s in summary]
        out.append("")

    if chapters:
        out += ["CHAPTERS", "-"]
        out += [f"{c.timestamp}  {c.title}" for c in chapters]
        out.append("")

    if quotes:
        out += ["QUOTES", "-"]
        out += [f"[{q.timestamp}] \u201c{q.text}\u201d" for q in quotes]
        out.append("")

    if keywords:
        out += ["TAGS", "-"]
        out.append(", ".join(k.term for k in keywords))
    return "\n".join(out)


# --- shorts -----------------------------------------------------------------

@dataclass(frozen=True)
class Short:
    """A vertical-clip candidate with the actual cut range."""

    start: float
    end: float
    hook: str
    reason: str
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def range_text(self) -> str:
        return f"{format_timestamp(self.start)} \u2192 {format_timestamp(self.end)}"

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "range": self.range_text,
                "duration": round(self.duration, 1), "hook": self.hook,
                "reason": self.reason, "score": round(self.score, 3)}


def build_shorts(
    transcript: Transcript,
    *,
    quotes: Sequence[Quote] = (),
    chapters: Sequence[Chapter] = (),
    min_seconds: float = 20.0,
    max_seconds: float = 75.0,
    n: int = 5,
) -> List[Short]:
    """Find 20–75 second segments that work as vertical clips.

    A quote alone is not a clip — it needs a little context before it and a
    beat after. This expands each quote outward to the target length, so the
    range you get back is something you can actually cut.
    """
    candidates: List[Short] = []
    total = transcript.total_duration

    for quote in quotes:
        # centre the clip on the quote, then expand to the target window
        start = max(0.0, quote.start - min_seconds * 0.4)
        end = min(total, quote.end + min_seconds * 0.4)
        if end - start < min_seconds:
            shortfall = min_seconds - (end - start)
            start = max(0.0, start - shortfall / 2)
            end = min(total, end + shortfall / 2)
        if end - start > max_seconds:
            end = start + max_seconds

        # A clip shorter than the minimum is not what was asked for. Allow it
        # only when the whole episode is shorter than the minimum — otherwise
        # the user gets a 16s "short" from a 60-minute episode.
        if end - start < min_seconds and total > min_seconds:
            continue

        candidates.append(Short(
            start=round(start, 2), end=round(end, 2),
            hook=quote.text[:120],
            reason=f"quote at {quote.timestamp} ({quote.duration:.0f}s)",
            score=quote.score))

    # chapters are a fallback when there are few quotable moments
    for chapter in chapters:
        if min_seconds <= chapter.duration <= max_seconds:
            candidates.append(Short(
                start=chapter.start, end=chapter.end,
                hook=chapter.title,
                reason=f"self-contained chapter at {chapter.timestamp}",
                score=1.0 + len(chapter.keywords) * 0.1))

    candidates.sort(key=lambda s: -s.score)

    # drop overlapping candidates — two clips of the same 40 seconds are one clip
    out: List[Short] = []
    for candidate in candidates:
        if any(not (candidate.end < kept.start or candidate.start > kept.end)
               for kept in out):
            continue
        out.append(candidate)
        if len(out) >= n:
            break
    return out


# --- SEO --------------------------------------------------------------------

@dataclass(frozen=True)
class SEO:
    title: str
    description: str
    hashtags: List[str]
    keywords: List[str]

    def to_dict(self) -> dict:
        return {"title": self.title, "description": self.description,
                "hashtags": list(self.hashtags),
                "keywords": list(self.keywords),
                "title_length": len(self.title),
                "description_length": len(self.description)}


def build_seo(
    transcript: Transcript,
    *,
    title: str = "",
    summary: Sequence[Sentence] = (),
    hashtag_prefix: str = "#",
    max_hashtags: int = 8,
) -> SEO:
    """Titles, description and tags that fit the real field limits.

    YouTube truncates titles past ~70 characters and descriptions in search
    results past ~155. A title that looks fine in your editor gets cut on the
    results page, which is the whole reason for the limits here.
    """
    keywords = top_keywords(transcript.text, n=12)

    heading = title or transcript.title
    if not heading:
        heading = " ".join(k.term.capitalize() for k in keywords[:3]) or "Episode"
    if len(heading) > TITLE_LIMIT:
        heading = heading[: TITLE_LIMIT - 3].rstrip() + "..."

    description = " ".join(s.text for s in summary[:2])
    if not description:
        description = (f"{format_timestamp(transcript.total_duration)} episode on "
                       + ", ".join(k.term for k in keywords[:4]) + ".")
    if len(description) > META_LIMIT:
        description = description[: META_LIMIT - 3].rstrip() + "..."

    tags: List[str] = []
    for keyword in keywords[:max_hashtags]:
        tag = "#" + "".join(keyword.term.split())
        if tag.lower() not in [t.lower() for t in tags]:
            tags.append(tag)

    return SEO(
        title=heading, description=description, hashtags=tags,
        keywords=[k.term for k in keywords])
