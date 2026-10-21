"""``bolkhabar`` command line.

One transcript in, a week of content out. No API key, no model, no network.

Usage::

    bolkhabar all examples/episode.json
    bolkhabar thread examples/episode.json
    bolkhabar blog examples/episode.json --title "My episode"
    bolkhabar newsletter examples/episode.json
    bolkhabar notes examples/episode.json
    bolkhabar shorts examples/episode.json
    bolkhabar seo examples/episode.json
    bolkhabar chapters examples/episode.json

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .core.analysis import extract_quotes, segment_chapters, summarize, top_keywords
from .core.transcript import Transcript, TranscriptError, format_timestamp
from .core.writers import (
    build_blog,
    build_newsletter,
    build_seo,
    build_shorts,
    build_show_notes,
    build_thread,
)

__all__ = ["main", "build_parser", "analyse"]


def analyse(transcript: Transcript) -> dict:
    """Run every analysis once and share the results.

    Done once and passed around rather than recomputed per writer, because the
    chapters a blog post uses must be the same chapters the thread uses —
    otherwise the two pieces of content disagree about the episode.
    """
    chapters = segment_chapters(transcript)
    quotes = extract_quotes(transcript)
    summary = summarize(transcript)
    return {
        "chapters": chapters, "quotes": quotes, "summary": summary,
        "keywords": top_keywords(transcript.text, n=12),
    }


# --- subcommands ------------------------------------------------------------

def _load(args) -> Tuple_transcript:
    return Transcript.from_file(args.transcript, title=getattr(args, "title", ""))


Tuple_transcript = Transcript  # readable alias for the annotation above


def cmd_chapters(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    chapters = segment_chapters(transcript)
    if args.json:
        print(json.dumps([c.to_dict() for c in chapters],
                         ensure_ascii=False, indent=2), file=out)
        return 0
    print(f"{len(chapters)} chapters over "
          f"{format_timestamp(transcript.total_duration)}\n", file=out)
    for chapter in chapters:
        print(f"{chapter.timestamp}  {chapter.title}", file=out)
        if chapter.keywords:
            print(f"           {', '.join(k.term for k in chapter.keywords)}", file=out)
    return 0


def cmd_thread(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    tweets = build_thread(
        transcript, chapters=analysis["chapters"], quotes=analysis["quotes"],
        hook=args.hook, cta=args.cta, limit=args.limit,
        max_tweets=args.max_tweets)
    for tweet in tweets:
        print(tweet, file=out)
        print(f"\n{'-' * 60}  ({len(tweet)} chars)\n", file=out)
    over = [t for t in tweets if len(t) > args.limit]
    if over:
        print(f"WARNING: {len(over)} tweet(s) exceed {args.limit} characters",
              file=sys.stderr)
        return 1
    return 0


def cmd_blog(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    post = build_blog(transcript, chapters=analysis["chapters"],
                      quotes=analysis["quotes"], summary=analysis["summary"],
                      title=args.title)
    if args.json:
        print(json.dumps(post.to_dict(), ensure_ascii=False, indent=2), file=out)
        return 0
    print(post.body, file=out)
    print(f"\n---\n{post.word_count} words, ~{post.read_minutes} min read",
          file=out)
    return 0


def cmd_newsletter(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    payload = build_newsletter(
        transcript, chapters=analysis["chapters"], quotes=analysis["quotes"],
        summary=analysis["summary"], title=args.title, listen_url=args.url)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
        return 0
    print(f"Subject : {payload['subject']}", file=out)
    print(f"Preview : {payload['preview']}", file=out)
    print(f"Duration: {payload['duration']}\n", file=out)
    if payload["takeaways"]:
        print("Takeaways:", file=out)
        print("\n".join(f"  \u2022 {t}" for t in payload["takeaways"]), file=out)
        print("", file=out)
    for section in payload["sections"]:
        print(f"[{section['timestamp']}] {section['heading']}", file=out)
        print(f"  {section['text'][:200]}\n", file=out)
    return 0


def cmd_notes(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    print(build_show_notes(
        transcript, chapters=analysis["chapters"], quotes=analysis["quotes"],
        summary=analysis["summary"], title=args.title,
        guests=[g for g in args.guests.split(",") if g] if args.guests else ()),
        file=out)
    return 0


def cmd_shorts(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    shorts = build_shorts(transcript, quotes=analysis["quotes"],
                          chapters=analysis["chapters"], n=args.n)
    if args.json:
        print(json.dumps([s.to_dict() for s in shorts],
                         ensure_ascii=False, indent=2), file=out)
        return 0
    if not shorts:
        print("no clip candidates found", file=out)
        return 0
    for index, short in enumerate(shorts, start=1):
        print(f"{index}. {short.range_text}  ({short.duration:.0f}s)", file=out)
        print(f"   hook: {short.hook}", file=out)
        print(f"   why:  {short.reason}\n", file=out)
    return 0


def cmd_seo(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    seo = build_seo(transcript, title=args.title, summary=analysis["summary"])
    if args.json:
        print(json.dumps(seo.to_dict(), ensure_ascii=False, indent=2), file=out)
        return 0
    print(f"Title       ({len(seo.title)}/60): {seo.title}", file=out)
    print(f"Description ({len(seo.description)}/155): {seo.description}", file=out)
    print(f"Hashtags    : {' '.join(seo.hashtags)}", file=out)
    print(f"Keywords    : {', '.join(seo.keywords)}", file=out)
    return 0


def cmd_all(args, out) -> int:
    transcript = Transcript.from_file(args.transcript, title=args.title)
    analysis = analyse(transcript)
    duration = format_timestamp(transcript.total_duration)

    print("=" * 62, file=out)
    print(f"{args.title or transcript.title or 'Episode'}  ({duration})", file=out)
    print("=" * 62, file=out)

    print(f"\n{len(analysis['chapters'])} chapters, "
          f"{len(analysis['quotes'])} quotes, "
          f"{len(analysis['summary'])} summary lines\n", file=out)

    print("CHAPTERS", file=out)
    for chapter in analysis["chapters"]:
        print(f"  {chapter.timestamp}  {chapter.title}", file=out)

    print("\nBEST QUOTES", file=out)
    for quote in analysis["quotes"]:
        print(f"  [{quote.timestamp}] \u201c{quote.text[:100]}\u201d", file=out)
        print(f"       {', '.join(quote.reasons)}", file=out)

    print("\nSHOW NOTES", file=out)
    print(build_show_notes(transcript, chapters=analysis["chapters"],
                           quotes=analysis["quotes"],
                           summary=analysis["summary"], title=args.title),
          file=out)

    print("\n\nX THREAD", file=out)
    for tweet in build_thread(transcript, chapters=analysis["chapters"],
                              quotes=analysis["quotes"]):
        print(f"  {tweet}", file=out)

    print("\nSEO", file=out)
    seo = build_seo(transcript, title=args.title, summary=analysis["summary"])
    print(f"  Title: {seo.title}", file=out)
    print(f"  Desc:  {seo.description}", file=out)
    print(f"  Tags:  {' '.join(seo.hashtags)}", file=out)

    print("\nSHORTS", file=out)
    for short in build_shorts(transcript, quotes=analysis["quotes"],
                              chapters=analysis["chapters"]):
        print(f"  {short.range_text}  ({short.duration:.0f}s) — {short.hook[:60]}",
              file=out)
    return 0


# --- wiring -----------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("transcript", help="path to a word-level transcript JSON")
    parser.add_argument("--title", default="", help="episode title")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bolkhabar",
        description="Turn one podcast transcript into a thread, a blog post, a "
                    "newsletter, show notes, shorts and SEO. Offline, free, "
                    "zero dependencies.",
    )
    parser.add_argument("--version", action="version", version="bolkhabar 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("all", help="everything at once")
    _add_common(p)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("chapters", help="topic segments with timestamps")
    _add_common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_chapters)

    p = sub.add_parser("thread", help="an X/Twitter thread")
    _add_common(p)
    p.add_argument("--hook", default="", help="the first tweet")
    p.add_argument("--cta", default="", help="the closing tweet")
    p.add_argument("--limit", type=int, default=280)
    p.add_argument("--max-tweets", type=int, default=15)
    p.set_defaults(func=cmd_thread)

    p = sub.add_parser("blog", help="a blog post")
    _add_common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_blog)

    p = sub.add_parser("newsletter", help="a newsletter payload")
    _add_common(p)
    p.add_argument("--url", default="", help="listen link")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_newsletter)

    p = sub.add_parser("notes", help="timestamped show notes")
    _add_common(p)
    p.add_argument("--guests", default="", help="comma-separated")
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("shorts", help="vertical clip candidates")
    _add_common(p)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_shorts)

    p = sub.add_parser("seo", help="title, description, tags")
    _add_common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_seo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, sys.stdout)
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
