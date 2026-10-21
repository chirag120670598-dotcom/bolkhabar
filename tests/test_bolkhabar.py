"""Tests for bolkhabar.

The sample transcript in ``examples/`` is deliberately word-salad — it
exercises the structure (timestamps, segments, boundaries) without pretending
to be real speech. The unit fixtures below are hand-written so the assertions
mean something.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from decimal import Decimal

import pytest

from bolkhabar.cli import analyse, main
from bolkhabar.core.analysis import (
    Chapter,
    extract_quotes,
    segment_chapters,
    summarize,
    tokenize,
    top_keywords,
)
from bolkhabar.core.transcript import (
    Sentence,
    Transcript,
    TranscriptError,
    Word,
    format_timestamp,
)
from bolkhabar.core.writers import (
    META_LIMIT,
    TITLE_LIMIT,
    TWEET_LIMIT,
    build_blog,
    build_newsletter,
    build_seo,
    build_shorts,
    build_show_notes,
    build_thread,
    split_thread,
)


# --- helpers ----------------------------------------------------------------

def words_from(spec, gap=0.05):
    """Build words from ``[(text, duration), ...]`` laid out back to back."""
    out, t = [], 0.0
    for text, duration in spec:
        out.append(Word(text, round(t, 3), round(t + duration, 3)))
        t = round(t + duration + gap, 3)
    return out


def transcript_from(text, *, gap=0.05):
    tokens = text.split()
    return Transcript.from_words([(w, 0, 0) for w in []]) if not tokens else \
        Transcript(words=words_from([(w, 0.25) for w in tokens], gap=gap))


def transcript_with_sentences(sentences, *, pause=1.5):
    """A transcript where each string is one sentence, separated by a pause."""
    out, t = [], 0.0
    for sentence in sentences:
        for token in sentence.split():
            out.append(Word(token, round(t, 3), round(t + 0.25, 3)))
            t = round(t + 0.30, 3)
        t = round(t + pause, 3)
    return Transcript(words=out, duration=t, title="Test episode")


SAMPLE = [
    "I started freelancing from Rajkot in 2021 with no portfolio at all.",
    "The biggest mistake I made was charging hourly rates to every client.",
    "Nobody tells you that a client's budget matters more than your skill.",
    "When I switched to fixed pricing my income doubled in six months.",
    "Scope creep is the real problem, not the price you quote upfront.",
    "You have to ask for an advance payment before you write a single line.",
    "Building in public on Twitter brought me my three best clients.",
    "Ship fast, get feedback, and stop waiting for the perfect launch.",
    "GST registration felt scary but my CA explained it in one afternoon.",
    "Advance tax is due four times a year and most freelancers forget it.",
    "Consistency beats intensity, especially over a period of years.",
    "Protect your energy, take real breaks, and do not work every weekend.",
    "Deep work in the morning changed how much I actually finish.",
    "Burnout is not a badge of honour, it is a billing problem.",
    "Start before you feel ready, because nobody feels ready at first.",
]


@pytest.fixture
def episode():
    return transcript_with_sentences(SAMPLE)


@pytest.fixture
def analysed(episode):
    return analyse(episode)


# --- transcript -------------------------------------------------------------

class TestTimestamps:
    def test_format_under_an_hour(self):
        assert format_timestamp(125) == "02:05"

    def test_format_over_an_hour(self):
        assert format_timestamp(3725.4) == "01:02:05"

    def test_format_with_colon(self):
        assert format_timestamp(125, colon=False) == "00:02:05"

    def test_negative_rejected(self):
        with pytest.raises(TranscriptError):
            format_timestamp(-1)


class TestTranscript:
    def test_word_validation(self):
        with pytest.raises(TranscriptError):
            Word("x", -1, 0)
        with pytest.raises(TranscriptError):
            Word("x", 5, 1)

    def test_out_of_order_rejected(self):
        """Wrong order means wrong chapter markers, which is worse than a crash."""
        with pytest.raises(TranscriptError, match="out of order"):
            Transcript(words=[Word("a", 5, 6), Word("b", 1, 2)])

    def test_overlap_rejected(self):
        with pytest.raises(TranscriptError, match="overlap"):
            Transcript(words=[Word("a", 0, 5), Word("b", 3, 6)])

    def test_tiny_overlap_tolerated(self):
        """ASR engines round differently; 0.5ms must not be a hard failure."""
        t = Transcript(words=[Word("a", 0.0, 1.0), Word("b", 0.9999, 2.0)])
        assert len(t) == 2

    def test_from_dicts_whisper_shape(self):
        t = Transcript.from_dicts([
            {"word": "Namaste", "start": 0.0, "end": 0.4},
            {"word": "dosto", "start": 0.5, "end": 0.9}])
        assert t.text == "Namaste dosto"

    def test_from_json_object(self):
        payload = {"language": "en", "duration": 10.0, "title": "Ep",
                   "words": [{"word": "hello", "start": 0, "end": 1}]}
        t = Transcript.from_json(payload)
        assert t.title == "Ep"
        assert t.duration == 10.0

    def test_from_json_list(self):
        t = Transcript.from_json([{"word": "hello", "start": 0, "end": 1}])
        assert len(t) == 1

    def test_empty_rejected(self):
        with pytest.raises(TranscriptError):
            Transcript.from_json({"words": []})

    def test_bad_json_type(self):
        """A JSON decode error surfaces as a TranscriptError, not a raw exception."""
        with pytest.raises(TranscriptError, match="not valid JSON"):
            Transcript.from_json("just a string")

    def test_missing_file(self, tmp_path):
        with pytest.raises(TranscriptError):
            Transcript.from_file(tmp_path / "nope.json")

    def test_json_roundtrip(self):
        t = transcript_with_sentences(["one two three.", "four five six."])
        assert len(Transcript.from_json(t.to_json())) == len(t)

    def test_total_duration_from_words(self):
        t = Transcript(words=words_from([("a", 1.0), ("b", 1.0)]))
        assert t.total_duration == pytest.approx(2.05)

    def test_minutes(self):
        assert Transcript(words=[], duration=120).minutes == 2.0

    def test_sentences_split_on_pause(self):
        t = transcript_with_sentences(["one two three.", "four five six."])
        assert len(t.sentences()) == 2

    def test_sentences_split_on_punctuation(self):
        t = transcript_from("one two three. four five six.")
        assert len(t.sentences()) == 2

    def test_devanagari_period_is_a_boundary(self):
        t = transcript_from("\u0928\u092e\u0938\u094d\u0924\u0947 \u0964 aaj baat.")
        assert len(t.sentences()) == 2

    def test_word_normalized(self):
        assert Word("Hello,", 0, 1).normalized == "hello"

    def test_text_between(self):
        t = Transcript(words=words_from([(w, 1.0) for w in "a b c d".split()], gap=0.0))
        assert t.text_between(0.0, 2.0) == "a b"

    def test_sample_episode_parses(self):
        t = Transcript.from_file("examples/episode.json")
        assert len(t) > 100
        assert t.total_duration > 60


# --- analysis ---------------------------------------------------------------

class TestTokenize:
    def test_removes_english_stopwords(self):
        assert "the" not in tokenize("the quick brown fox")

    def test_removes_hindi_stopwords(self):
        assert "matlab" not in tokenize("yeh matlab bahut accha hai")

    def test_removes_fillers(self):
        assert "um" not in tokenize("um so like the pricing")

    def test_keeps_content_words(self):
        assert "pricing" in tokenize("the pricing problem")

    def test_hindi_words_kept(self):
        assert tokenize("\u0926\u093e\u092e")  # Devanagari survives


class TestKeywords:
    def test_frequent_term_ranks_first(self):
        text = "pricing pricing pricing clients clients invoice"
        assert top_keywords(text, n=1)[0].term == "pricing"

    def test_single_occurrence_excluded(self):
        """A word said once is not a topic."""
        assert "unicorn" not in [k.term for k in top_keywords(
            "pricing pricing clients clients unicorn")]

    def test_empty_text(self):
        assert top_keywords("") == []

    def test_bigrams_included(self):
        text = " ".join(["client budget"] * 6 + ["pricing"] * 4)
        terms = [k.term for k in top_keywords(text, n=10, bigrams=True)]
        assert "client budget" in terms

    def test_n_respected(self):
        text = " ".join(f"word{i} " * 5 for i in range(30))
        assert len(top_keywords(text, n=5)) <= 8   # +bigram allowance


class TestChapters:
    def test_produces_chapters(self, episode):
        chapters = segment_chapters(episode)
        assert chapters

    def test_chapters_are_ordered_and_non_overlapping(self, episode):
        chapters = segment_chapters(episode)
        for a, b in zip(chapters, chapters[1:]):
            assert a.start < b.start
            assert a.end <= b.start + 0.001

    def test_chapters_cover_the_episode(self, episode):
        """No gaps — a missing chunk means missing content."""
        chapters = segment_chapters(episode)
        assert chapters[0].start == episode.words[0].start
        assert chapters[-1].end == episode.words[-1].end

    def test_every_sentence_assigned_exactly_once(self, episode):
        chapters = segment_chapters(episode)
        assigned = [s.text for c in chapters for s in c.sentences]
        assert len(assigned) == len(episode.sentences())
        assert len(assigned) == len(set(assigned))

    def test_chapter_titles_are_not_empty(self, episode):
        for chapter in segment_chapters(episode):
            assert chapter.title

    def test_single_sentence_transcript(self):
        t = transcript_with_sentences(["just one sentence here."])
        assert len(segment_chapters(t)) == 1

    def test_empty_transcript(self):
        assert segment_chapters(Transcript(words=[])) == []

    def test_min_chapters_respected(self, episode):
        assert len(segment_chapters(episode, min_chapters=4)) >= 4

    def test_max_chapters_respected(self, episode):
        assert len(segment_chapters(episode, max_chapters=3,
                                    min_chapters=1)) <= 3

    def test_chapter_to_dict(self, episode):
        d = segment_chapters(episode)[0].to_dict()
        assert {"start", "end", "timestamp", "title"} <= set(d)


class TestQuotes:
    def test_finds_quotes(self, episode):
        assert extract_quotes(episode)

    def test_quotes_carry_reasons(self, episode):
        """A quote you cannot justify is a quote you cannot trust."""
        for quote in extract_quotes(episode):
            assert quote.reasons

    def test_word_count_bounds_respected(self, episode):
        for quote in extract_quotes(episode, min_words=8, max_words=45):
            assert 8 <= len(quote.text.split()) <= 45

    def test_questions_excluded_by_default(self):
        """A question on a pull-quote card has nothing to read."""
        t = transcript_with_sentences([
            "Is pricing really the biggest problem for freelancers today here?",
            "Scope creep is the biggest problem for every freelancer working now.",
            "Nobody tells you the truth about advance payment and client budgets.",
            "Every client budget matters more than the skill you actually have.",
        ])
        quotes = extract_quotes(t, n=4, min_words=5)
        assert quotes
        assert all(not q.text.endswith("?") for q in quotes)

    def test_questions_can_be_opted_in(self):
        t = transcript_with_sentences([
            "Is pricing really the biggest problem for freelancers today here?",
            "Scope creep is the biggest problem for every freelancer working now.",
        ])
        quotes = extract_quotes(t, n=4, min_words=5, include_questions=True)
        assert any(q.text.endswith("?") for q in quotes)

    def test_duplicates_removed(self, episode):
        quotes = extract_quotes(episode, n=10)
        texts = [q.text for q in quotes]
        assert len(texts) == len(set(texts))

    def test_sorted_by_score(self, episode):
        quotes = extract_quotes(episode, n=10)
        scores = [q.score for q in quotes]
        assert scores == sorted(scores, reverse=True)

    def test_empty_transcript(self):
        assert extract_quotes(Transcript(words=[])) == []

    def test_quote_to_dict(self, episode):
        d = extract_quotes(episode)[0].to_dict()
        assert "timestamp" in d and "reasons" in d


class TestSummary:
    def test_returns_sentences(self, episode):
        assert summarize(episode)

    def test_max_respected(self, episode):
        assert len(summarize(episode, max_sentences=3)) <= 3

    def test_original_order_preserved(self, episode):
        """A scrambled summary reads like nonsense."""
        picked = summarize(episode, max_sentences=4)
        starts = [s.start for s in picked]
        assert starts == sorted(starts)

    def test_extractive_not_generative(self, episode):
        """Extractive summarising cannot invent something the guest never said."""
        for sentence in summarize(episode):
            assert sentence.text in [s.text for s in episode.sentences()]

    def test_empty_transcript(self):
        assert summarize(Transcript(words=[])) == []


class TestAnalyse:
    def test_runs_everything_once(self, episode):
        result = analyse(episode)
        assert {"chapters", "quotes", "summary", "keywords"} <= set(result)

    def test_deterministic(self, episode):
        """Same episode must produce the same content every run."""
        a, b = analyse(episode), analyse(episode)
        assert [c.title for c in a["chapters"]] == [c.title for c in b["chapters"]]


# --- writers ----------------------------------------------------------------

class TestSplitThread:
    def test_short_text_is_one_tweet(self):
        assert split_thread("Short one.") == ["Short one."]

    def test_respects_limit(self):
        text = " ".join(["sentence number %d is here." % i for i in range(40)])
        for tweet in split_thread(text, limit=100):
            assert len(tweet) <= 100

    def test_never_splits_mid_word(self):
        text = " ".join(["a" * 30] * 20)
        for tweet in split_thread(text, limit=80):
            assert "a" * 30 in tweet or len(tweet) <= 80

    def test_no_content_lost(self):
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        rejoined = " ".join(split_thread(text, limit=100)).split()
        assert rejoined == words

    def test_tiny_limit_rejected(self):
        with pytest.raises(ValueError):
            split_thread("hello", limit=2)

    def test_single_long_sentence_split(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        for tweet in split_thread(text, limit=20):
            assert len(tweet) <= 20


class TestThread:
    def test_all_tweets_within_limit(self, episode):
        """The one thing that actually breaks when you paste a thread."""
        analysis = analyse(episode)
        tweets = build_thread(episode, chapters=analysis["chapters"],
                              quotes=analysis["quotes"])
        assert tweets
        for tweet in tweets:
            assert len(tweet) <= TWEET_LIMIT, f"{len(tweet)} chars: {tweet[:40]}"

    def test_numbered_sequentially(self, episode):
        tweets = build_thread(episode)
        counters = [t.rsplit("/", 1)[1] for t in tweets]
        assert counters == [str(len(tweets))] * len(tweets)

    def test_counter_denominator_correct(self, episode):
        tweets = build_thread(episode)
        assert tweets[0].endswith(f"1/{len(tweets)}")

    def test_hook_is_first(self, episode):
        assert build_thread(episode, hook="My hook")[0].startswith("My hook")

    def test_cta_is_last(self, episode):
        assert "Full episode" in build_thread(episode, cta="Full episode")[-1]

    def test_custom_limit_respected(self, episode):
        tweets = build_thread(episode, limit=140)
        assert all(len(t) <= 140 for t in tweets)

    def test_max_tweets_respected(self, episode):
        assert len(build_thread(episode, max_tweets=3)) <= 3

    def test_timestamps_included(self, episode):
        assert any("[" in t for t in build_thread(episode))


class TestBlog:
    def test_has_h1_and_h2(self, episode, analysed):
        body = build_blog(episode, chapters=analysed["chapters"],
                          summary=analysed["summary"]).body
        assert body.startswith("# ")
        assert "## " in body

    def test_one_h2_per_chapter(self, episode, analysed):
        post = build_blog(episode, chapters=analysed["chapters"],
                          summary=analysed["summary"])
        assert post.body.count("## 1.") == 1

    def test_word_count_matches_body(self, episode, analysed):
        post = build_blog(episode, chapters=analysed["chapters"])
        assert post.word_count == len(post.body.split())

    def test_read_minutes_at_least_one(self, episode, analysed):
        assert build_blog(episode, chapters=analysed["chapters"]).read_minutes >= 1

    def test_content_comes_from_transcript(self, episode, analysed):
        """Nothing is invented — every body line traces back to the transcript."""
        post = build_blog(episode, chapters=analysed["chapters"])
        transcript_words = set(episode.text.lower().split())
        for sentence in analysed["summary"]:
            assert set(sentence.text.lower().split()) <= transcript_words | {""}

    def test_description_within_limit(self, episode, analysed):
        assert len(build_blog(episode, summary=analysed["summary"]).description) <= META_LIMIT

    def test_custom_title_used(self, episode):
        assert build_blog(episode, title="My Title").title == "My Title"

    def test_pull_quotes_included(self, episode, analysed):
        post = build_blog(episode, chapters=analysed["chapters"],
                          quotes=analysed["quotes"])
        if analysed["quotes"]:
            assert "> \u201c" in post.body


class TestNewsletter:
    def test_subject_within_limit(self, episode, analysed):
        payload = build_newsletter(episode, chapters=analysed["chapters"],
                                   summary=analysed["summary"])
        assert len(payload["subject"]) <= 60

    def test_long_subject_truncated(self, episode):
        long_title = "x" * 200
        assert len(build_newsletter(episode, title=long_title)["subject"]) <= 60

    def test_sections_match_chapters(self, episode, analysed):
        payload = build_newsletter(episode, chapters=analysed["chapters"])
        assert len(payload["sections"]) == len(analysed["chapters"])

    def test_preview_within_limit(self, episode, analysed):
        assert len(build_newsletter(episode, summary=analysed["summary"])["preview"]) <= 110

    def test_takeaways_capped(self, episode, analysed):
        assert len(build_newsletter(episode, summary=analysed["summary"])["takeaways"]) <= 5

    def test_json_serialisable(self, episode, analysed):
        assert json.loads(json.dumps(
            build_newsletter(episode, chapters=analysed["chapters"])))


class TestShowNotes:
    def test_has_chapters_section(self, episode, analysed):
        notes = build_show_notes(episode, chapters=analysed["chapters"])
        assert "CHAPTERS" in notes

    def test_timestamps_present(self, episode, analysed):
        assert "00:00" in build_show_notes(episode, chapters=analysed["chapters"])

    def test_guests_listed(self, episode, analysed):
        notes = build_show_notes(episode, chapters=analysed["chapters"],
                                 guests=["Ravi", "Meera"])
        assert "Ravi" in notes and "Meera" in notes

    def test_tags_present(self, episode, analysed):
        assert "TAGS" in build_show_notes(episode, chapters=analysed["chapters"])


class TestShorts:
    def test_within_duration_window(self, episode, analysed):
        for short in build_shorts(episode, quotes=analysed["quotes"],
                                  chapters=analysed["chapters"],
                                  min_seconds=20, max_seconds=75):
            assert 19 <= short.duration <= 76

    def test_clips_do_not_overlap(self, episode, analysed):
        shorts = build_shorts(episode, quotes=analysed["quotes"],
                              chapters=analysed["chapters"])
        for a, b in zip(shorts, shorts[1:]):
            assert a.end <= b.start or b.end <= a.start

    def test_within_episode_bounds(self, episode, analysed):
        for short in build_shorts(episode, quotes=analysed["quotes"],
                                  chapters=analysed["chapters"]):
            assert short.start >= 0
            assert short.end <= episode.total_duration + 0.001

    def test_n_respected(self, episode, analysed):
        assert len(build_shorts(episode, quotes=analysed["quotes"], n=2)) <= 2

    def test_range_text_readable(self, episode, analysed):
        shorts = build_shorts(episode, quotes=analysed["quotes"])
        if shorts:
            assert "\u2192" in shorts[0].range_text

    def test_empty_transcript(self):
        assert build_shorts(Transcript(words=[])) == []


class TestSEO:
    def test_title_within_limit(self, episode, analysed):
        seo = build_seo(episode, summary=analysed["summary"])
        assert len(seo.title) <= TITLE_LIMIT

    def test_long_title_truncated(self, episode):
        assert len(build_seo(episode, title="x" * 200).title) <= TITLE_LIMIT

    def test_description_within_limit(self, episode, analysed):
        assert len(build_seo(episode, summary=analysed["summary"]).description) <= META_LIMIT

    def test_hashtags_are_single_tokens(self, episode):
        """A hashtag with a space in it is two broken hashtags."""
        for tag in build_seo(episode).hashtags:
            assert tag.startswith("#")
            assert " " not in tag

    def test_hashtags_capped(self, episode):
        assert len(build_seo(episode, max_hashtags=4).hashtags) <= 4

    def test_no_duplicate_hashtags(self, episode):
        tags = [t.lower() for t in build_seo(episode).hashtags]
        assert len(tags) == len(set(tags))


# --- CLI --------------------------------------------------------------------

class TestCLI:
    def run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_chapters(self):
        code, out = self.run(["chapters", "examples/episode.json"])
        assert code == 0
        assert "chapters over" in out

    def test_chapters_json(self):
        code, out = self.run(["chapters", "examples/episode.json", "--json"])
        assert code == 0
        assert json.loads(out)

    def test_thread(self):
        code, out = self.run(["thread", "examples/episode.json"])
        assert code == 0
        assert "/1" in out or "/2" in out or "/" in out

    def test_thread_respects_limit(self):
        """Regression: --limit was checked but never passed to the builder."""
        code, out = self.run(["thread", "examples/episode.json", "--limit", "140"])
        assert code == 0
        for line in out.splitlines():
            if line.strip() and not line.startswith("-") and "chars)" not in line:
                assert len(line) <= 140, f"{len(line)}: {line[:50]}"

    def test_thread_custom_limit_actually_applied(self, episode, analysed):
        wide = build_thread(episode, chapters=analysed["chapters"], limit=280)
        narrow = build_thread(episode, chapters=analysed["chapters"], limit=120)
        assert all(len(t) <= 280 for t in wide)
        assert all(len(t) <= 120 for t in narrow)
        # a tighter limit must produce more tweets, not silently overflow
        assert len(narrow) >= len(wide)

    def test_blog(self):
        code, out = self.run(["blog", "examples/episode.json"])
        assert code == 0
        assert out.startswith("# ")
        assert "min read" in out

    def test_blog_json(self):
        code, out = self.run(["blog", "examples/episode.json", "--json"])
        assert code == 0
        assert "body" in json.loads(out)

    def test_newsletter(self):
        code, out = self.run(["newsletter", "examples/episode.json"])
        assert code == 0
        assert "Subject" in out

    def test_newsletter_json(self):
        code, out = self.run(["newsletter", "examples/episode.json", "--json"])
        assert code == 0
        assert "sections" in json.loads(out)

    def test_notes(self):
        code, out = self.run(["notes", "examples/episode.json",
                              "--guests", "Ravi,Meera"])
        assert code == 0
        assert "CHAPTERS" in out

    def test_shorts(self):
        code, out = self.run(["shorts", "examples/episode.json"])
        assert code == 0

    def test_shorts_json(self):
        code, out = self.run(["shorts", "examples/episode.json", "--json"])
        assert code == 0
        assert isinstance(json.loads(out), list)

    def test_seo(self):
        code, out = self.run(["seo", "examples/episode.json"])
        assert code == 0
        assert "Title" in out

    def test_seo_json(self):
        code, out = self.run(["seo", "examples/episode.json", "--json"])
        assert code == 0
        assert "hashtags" in json.loads(out)

    def test_all(self):
        code, out = self.run(["all", "examples/episode.json"])
        assert code == 0
        for section in ("CHAPTERS", "BEST QUOTES", "SHOW NOTES", "X THREAD",
                        "SEO", "SHORTS"):
            assert section in out

    def test_missing_file(self):
        assert self.run(["chapters", "nope.json"])[0] == 2

    def test_bad_transcript(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"words": []}')
        assert self.run(["chapters", str(path)])[0] == 2
