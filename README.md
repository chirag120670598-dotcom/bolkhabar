<div align="center">

# 🎙️ bolkhabar

### *"Ek baar bolo, poora hafte bhar ka content."*

**One podcast transcript in. A thread, a blog post, a newsletter, show notes,
shorts and SEO out.**

Offline · free · deterministic · **zero dependencies** · MIT

[![CI](https://github.com/chirag120670598-dotcom/bolkhabar/actions/workflows/ci.yml/badge.svg)](https://github.com/chirag120670598-dotcom/bolkhabar/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![Tests](https://img.shields.io/badge/tests-120%20passing-success)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Try it in 30 seconds

```bash
git clone https://github.com/chirag120670598-dotcom/bolkhabar
cd bolkhabar

python -m bolkhabar.cli all examples/episode.json
```

One command prints chapters, best quotes, show notes, an X thread, SEO and
short-clip ranges for the whole episode.

```
CHAPTERS
  00:00  Client Budget Scope Creep Cold Email
  03:40  Build Public Ship Fast Indie Hacker
  05:30  Long Term Deep Work Advance Tax

BEST QUOTES
  [00:33] "Nobody tells you that a client's budget matters more..."
       2 keyword(s), strong claim, clean ending

X THREAD
  1. Client Budget Scope Creep Cold Email [00:00]  1/5
  ...

SEO
  Title (38/60): Client budget Scope creep Build public
  Desc  (155/155): ...
  Tags: #clientbudget #scopecreep #buildpublic

SHORTS
  00:25 → 00:46  (21s) — hook + actual clip range
```

## Commands

| Command | Output |
|---|---|
| `all` | everything, in one pass |
| `chapters` | topic segments with timestamps |
| `thread` | an X thread, every tweet guaranteed under the limit |
| `blog` | Markdown with one H2 per chapter and pull quotes |
| `newsletter` | subject, preview, sections, takeaways (JSON or text) |
| `notes` | timestamped show notes for the description box |
| `shorts` | 20–75 s vertical-clip candidates with real cut ranges |
| `seo` | title ≤60, description ≤155, hashtags |

Every command takes `--json`.

---

## Why no LLM

This is the controversial bit, so here is the reasoning.

| | LLM repurposer | **bolkhabar** |
|---|---|---|
| Cost | ~₹2–10 per episode | **₹0** |
| Speed | 30–90 s | **under a second** |
| Determinism | different output each run | **same episode → same output** |
| Can it invent a quote? | yes | **structurally impossible** |
| Works offline | no | **yes** |

That last row is the one that matters for a podcast. A summariser that
paraphrases can put words in your guest's mouth, and a hallucinated quote gets
screenshotted. `bolkhabar` is **extractive**: every sentence in the blog post,
newsletter and summary is a real sentence from the transcript, verbatim. There
is a test that asserts this.

The trade-off is honest: chapter *titles* come out as keyword phrases
("Client Budget Scope Creep Cold Email"), not prose. Rewrite those by hand, or
plug an LLM in at the adapter layer to polish them. The structure, timestamps
and quotes are what take the time, and those are exact.

### The platform limits are the feature

```
Title       (38/60):  Client budget Scope creep Build public
Description (155/155): ...
```

YouTube truncates titles past ~70 characters and descriptions in search results
past ~155. X truncates at 280. `bolkhabar` fits the real fields and tells you
the count. There is a test asserting **every tweet in a thread is within the
limit including its `1/N` counter** — because that is precisely the thing that
breaks when you paste a generated thread.

---

## Install

```bash
pip install bolkhabar
```

Or run from a clone. Nothing to build.

## Input format

Any word-level transcript JSON — the shape Whisper, faster-whisper and
[bolcut](https://github.com/chirag120670598-dotcom/bolcut) all produce:

```json
{
  "title": "Freelancing in India — the honest version",
  "language": "en",
  "duration": 535.0,
  "words": [
    { "word": "Namaste", "start": 0.0, "end": 0.4 },
    { "word": "dosto",   "start": 0.5, "end": 0.9 }
  ]
}
```

From an audio file:

```bash
pip install faster-whisper
python -c "
from faster_whisper import WhisperModel
import json, sys
model = WhisperModel('small')
segments, _ = model.transcribe('episode.mp3', word_timestamps=True)
words = [{'word': w.word, 'start': w.start, 'end': w.end}
         for s in segments for w in s.words]
json.dump({'words': words}, open('episode.json', 'w'))
"
bolkhabar all episode.json
```

Or use `bolcut transcribe episode.mp3 > episode.json` — same format.

## Usage

```bash
bolkhabar all episode.json --title "Ep 12: Pricing"

bolkhabar thread episode.json \
  --hook "I spent 3 years freelancing from a tier-2 city." \
  --cta "Full episode: link in bio" \
  --limit 280

bolkhabar blog episode.json --title "How I priced my way out of ₹15/hour" > post.md

bolkhabar newsletter episode.json --url https://example.com/ep12 --json > news.json

bolkhabar notes episode.json --guests "Ravi,Meera"

bolkhabar shorts episode.json --json
```

### As a library

```python
from bolkhabar.core.transcript import Transcript
from bolkhabar.cli import analyse
from bolkhabar.core.writers import build_thread, build_blog

t = Transcript.from_file("episode.json", title="Ep 12")
a = analyse(t)          # chapters + quotes + summary, computed once
print(build_thread(t, chapters=a["chapters"], quotes=a["quotes"]))
print(build_blog(t, chapters=a["chapters"], summary=a["summary"]).body)
```

`analyse()` runs once and the result is shared, so the chapters in your blog
post are the same chapters in your thread. Recomputing per writer is how two
pieces of content end up disagreeing about the same episode.

---

## How it works

### Chapters

Boundaries go where **vocabulary shift × pause length** peaks — the same
signal a listener uses to notice "ok, new topic". 70% weight on the vocabulary
change between adjacent sentences, 30% on the silence, with a minimum spacing
constraint so chapters do not cluster.

Chapters never overlap and never leave gaps. There is a test asserting that
every sentence is assigned to exactly one chapter — a missing chunk means
missing content, and that failure is silent.

### Quotes

Every quote carries **the reasons it was picked**:

```
[00:33] "Nobody tells you that a client's budget matters more than your skill."
     2 keyword(s), strong claim, contains a number, clean ending
```

Scored on: contains episode keywords · makes a strong claim · contains a
number · ends cleanly · sits mid-episode (intros are not the substance).
Penalised for filler. **Questions are excluded by default** — a question on a
pull-quote card has nothing to read — with `include_questions=True` to opt in.

### Shorts

A quote alone is not a clip. `build_shorts` expands each quote outward to the
20–75 s window and returns the actual cut range, so you get something you can
put straight into an editor. Candidates that cannot reach the minimum length
inside the episode are dropped rather than shipped short, and overlapping
candidates are deduplicated — two clips of the same 40 seconds are one clip.

### Keywords

TF scaled by how *spread out* a term is. A word repeated in one burst is a
tangent; a word spread across the episode is the topic. Bigrams are scored
separately and weighted up. Single-occurrence words are excluded — a word said
once is not a topic.

Hinglish is handled by stopword and filler lists in both scripts, plus
Devanagari-range preservation. `।` (danda) is treated as a sentence boundary.

---

## Tests

```bash
pip install -e ".[dev]"
pytest            # 120 tests, ~0.5s
```

Locked down:

- **every tweet fits the limit including its counter** — regression test for a
  real bug where the counter overflowed past 9 tweets
- **`--limit` is actually applied** — regression test for a real bug where the
  CLI checked the limit but never passed it to the builder
- **no content is lost when splitting** — rejoined tweets equal the input words
- **chapters never overlap and never leave gaps**
- **every sentence is assigned exactly once**
- **the summary is extractive** — every line exists verbatim in the transcript
- **hashtags contain no spaces** (a space means two broken hashtags)
- **titles ≤60, descriptions ≤155**
- **out-of-order and overlapping words are rejected** at load
- **0.5 ms overlaps are tolerated** — ASR engines round differently
- **the analysis is deterministic** — same episode, same output

---

## What is NOT here yet (honest roadmap)

| | |
|---|---|
| **ASR** | Brings a transcript, does not make one. Use Whisper or `bolcut`. |
| **LLM polish** | No adapter wired up. Titles are keyword phrases, not prose. |
| **Video cutting** | Returns clip *ranges*; it does not cut files. Pipe them into `bolcut`. |
| **Publishing** | No posting to X, LinkedIn or a CMS. Output is text and JSON. |
| **Hindi/Devanagari output** | Handles Hinglish input; does not translate. |
| **Speaker diarisation** | Single-speaker. Quotes are not attributed. |

## Roadmap

- [ ] LLM polish adapter (titles, hooks, tweet rewrites)
- [ ] Speaker diarisation and attributed quotes
- [ ] Hindi and English output variants from one recording
- [ ] LinkedIn and Instagram caption writers
- [ ] `--publish` for X and Ghost
- [ ] YouTube description with auto-chapters

## Contributing

The highest-value contributions are **better Hinglish stopword and filler
lists**, and **writers for platforms not covered yet**. Over-filtering Hinglish
is worse than under-filtering — a word wrongly removed disappears from every
chapter title and quote score — so add words with a test.

## License

MIT.

---

<div align="center">

**Made in Rajkot, Gujarat 🇮🇳** — *bol* (speak) + *khabar* (news). Say it once.

</div>
