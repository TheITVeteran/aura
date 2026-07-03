"""tools/knowledge_substrate/ingest_wikipedia.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stream a Wikipedia pages-articles dump (.xml.bz2) into the local corpus.

One command, bounded, resumable:

    .venv/bin/python tools/knowledge_substrate/ingest_wikipedia.py \
        --dump ~/.aura/knowledge/corpora/enwiki-latest-pages-articles-multistream.xml.bz2 \
        [--max-pages N] [--deadline-minutes M] [--resume]

- streams bz2 → xml.etree.iterparse: the ~100GB decompressed stream is
  never materialized; memory stays flat (elements cleared as consumed)
- namespace-0 articles only; redirects skipped
- wikitext cleaned with a deliberately simple, honest pass (templates,
  tables, refs, markup) — BM25 retrieval tolerates residual markup, and
  no cleaner short of a full parser is lossless; snippets may show
  occasional artifacts
- resumable: pages_processed persists in corpus meta; --resume fast-skips
  that many pages (parse-only) before inserting again
- bounded: --max-pages and --deadline-minutes both stop the run cleanly,
  flushing the current batch and recording progress
"""
from __future__ import annotations

import argparse
import bz2
import logging
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.knowledge.local_corpus import LocalCorpusStore, default_corpus_db_path

logger = logging.getLogger("Aura.Knowledge.WikipediaIngest")

_BATCH_SIZE = 400
_MIN_BODY_CHARS = 200          # stubs below this add index noise, not knowledge
_MAX_BODY_CHARS = 60_000       # cap pathological pages; FTS rows stay bounded

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.IGNORECASE)
_PIPED_LINK_RE = re.compile(r"\[\[[^\]|]*\|([^\]]+)\]\]")
_PLAIN_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_EXT_LINK_RE = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
_BARE_EXT_LINK_RE = re.compile(r"\[https?://[^\]]+\]")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"'{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def strip_templates(text: str) -> str:
    """Remove {{...}} template calls, handling nesting iteratively."""
    # Innermost-first passes; bounded iterations guard against pathology.
    pattern = re.compile(r"\{\{[^{}]*\}\}")
    for _ in range(12):
        text, n = pattern.subn("", text)
        if n == 0:
            break
    return text


def strip_tables(text: str) -> str:
    """Remove {| ... |} wiki tables (non-nested pass, iterated)."""
    pattern = re.compile(r"\{\|.*?\|\}", re.DOTALL)
    for _ in range(6):
        text, n = pattern.subn("", text)
        if n == 0:
            break
    return text


def clean_wikitext(raw: str) -> str:
    """Reduce wikitext to plain-ish prose for BM25 indexing."""
    text = _COMMENT_RE.sub("", raw)
    text = _REF_RE.sub("", text)
    text = strip_templates(text)
    text = strip_tables(text)
    text = _FILE_LINK_RE.sub("", text)
    text = _PIPED_LINK_RE.sub(r"\1", text)
    text = _PLAIN_LINK_RE.sub(r"\1", text)
    text = _EXT_LINK_RE.sub(r"\1", text)
    text = _BARE_EXT_LINK_RE.sub("", text)
    text = _HEADING_RE.sub(r"\1.", text)
    text = _TAG_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def iter_articles(dump_path: Path):
    """Yield (title, wikitext) for namespace-0, non-redirect pages."""
    stream = bz2.BZ2File(dump_path)
    context = ElementTree.iterparse(stream, events=("end",))
    for _event, elem in context:
        if not elem.tag.endswith("}page"):
            continue
        ns = elem.tag[: elem.tag.rindex("}") + 1]
        try:
            if (elem.findtext(f"{ns}ns") or "").strip() != "0":
                continue
            if elem.find(f"{ns}redirect") is not None:
                continue
            title = (elem.findtext(f"{ns}title") or "").strip()
            text = elem.findtext(f"{ns}revision/{ns}text") or ""
            if title and text:
                yield title, text
        finally:
            elem.clear()


def ingest(
    dump_path: Path,
    *,
    db_path: Path | None = None,
    max_pages: int = 0,
    deadline_minutes: float = 0.0,
    resume: bool = False,
    source_label: str = "wikipedia",
) -> dict:
    """Run the bounded, resumable ingest. Returns a summary dict."""
    store = LocalCorpusStore(db_path or default_corpus_db_path())
    skip = 0
    if resume:
        skip = int(store.get_meta(f"{source_label}_pages_processed", "0") or 0)
        if skip:
            logger.info("Resuming: fast-skipping %d already-processed pages", skip)

    deadline = time.monotonic() + deadline_minutes * 60 if deadline_minutes > 0 else None
    processed = skip
    inserted_total = 0
    skipped_short = 0
    batch: list[tuple[str, str, str]] = []
    started = time.monotonic()
    stop_reason = "dump_exhausted"

    def flush() -> None:
        nonlocal inserted_total, batch
        if batch:
            inserted_total += store.add_documents(batch)
            batch = []
        store.set_meta(f"{source_label}_pages_processed", str(processed))
        store.set_meta("sources", source_label)

    try:
        for index, (title, wikitext) in enumerate(iter_articles(dump_path)):
            if index < skip:
                continue
            processed = index + 1
            body = clean_wikitext(wikitext)
            if len(body) < _MIN_BODY_CHARS:
                skipped_short += 1
            else:
                batch.append((title, body[:_MAX_BODY_CHARS], source_label))
                if len(batch) >= _BATCH_SIZE:
                    flush()
            if max_pages and (processed - skip) >= max_pages:
                stop_reason = "max_pages"
                break
            if deadline and time.monotonic() > deadline:
                stop_reason = "deadline"
                break
            if processed % 20_000 == 0:
                elapsed = time.monotonic() - started
                logger.info(
                    "ingest progress: %d pages, %d indexed, %.0fs elapsed",
                    processed, inserted_total, elapsed,
                )
    finally:
        flush()

    return {
        "pages_processed": processed,
        "documents_indexed": inserted_total,
        "skipped_short": skipped_short,
        "stop_reason": stop_reason,
        "elapsed_s": round(time.monotonic() - started, 1),
        "corpus": store.status(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--deadline-minutes", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.dump.exists():
        logger.error("dump not found: %s", args.dump)
        return 2
    summary = ingest(
        args.dump,
        db_path=args.db,
        max_pages=args.max_pages,
        deadline_minutes=args.deadline_minutes,
        resume=args.resume,
    )
    logger.info("ingest summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
