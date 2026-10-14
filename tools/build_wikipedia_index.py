#!/usr/bin/env python3
"""Build a full-text index of English Wikipedia that her runtime can search without a network or a model.

The source is the ``wikimedia/wikipedia`` 20231101.en snapshot (parquet, one
row per article). A passage is a paragraph as Wikipedia's editors wrote it:
the article text split at its blank lines, as the snapshot marks paragraphs
(a single line break falls inside a list or a table), empty ones dropped. Each
passage is stored with its article's title in an SQLite FTS5 table (Porter
stemming, BM25 ranking), which SQLite builds and searches from disk, so the
index costs little memory beside her cortex.

The build is resumable: each parquet file is committed in one transaction
and recorded, and a file already recorded is skipped.

Usage:
    build_wikipedia_index.py --source DIR --index FILE [--limit-files N]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
    "title, body, article UNINDEXED, paragraph UNINDEXED, tokenize='porter unicode61')",
    "CREATE TABLE IF NOT EXISTS built (file TEXT PRIMARY KEY, articles INTEGER, passages INTEGER, seconds REAL)",
)


def paragraphs(text: str) -> list[str]:
    """An article's paragraphs: its blocks between blank lines, stripped, empty ones dropped."""
    return [block.strip() for block in re.split(r"\n\s*\n", str(text or "")) if block.strip()]


def index_file(connection: sqlite3.Connection, path: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq

    articles = passages = 0
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["id", "title", "text"]):
        for article in batch.to_pylist():
            articles += 1
            for number, paragraph in enumerate(paragraphs(article["text"])):
                rows.append((article["title"], paragraph, article["id"], number))
        passages += len(rows)
        connection.executemany("INSERT INTO passages(title, body, article, paragraph) VALUES (?, ?, ?, ?)", rows)
        rows = []
    return articles, passages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="directory of the snapshot's parquet files")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.source.expanduser().glob("*.parquet"))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        print("no parquet files", file=sys.stderr)
        return 1
    index = args.index.expanduser()
    index.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index)
    connection.execute("PRAGMA journal_mode=WAL")
    for statement in SCHEMA:
        connection.execute(statement)
    done = {row[0] for row in connection.execute("SELECT file FROM built")}
    for path in files:
        if path.name in done:
            continue
        started = time.monotonic()
        with connection:
            articles, passages = index_file(connection, path)
            connection.execute("INSERT INTO built VALUES (?, ?, ?, ?)",
                               (path.name, articles, passages, round(time.monotonic() - started, 1)))
        print(f"{path.name}: {articles} articles, {passages} passages, {time.monotonic() - started:.0f} s", flush=True)
    with connection:
        connection.execute("INSERT INTO passages(passages) VALUES ('optimize')")
    totals = connection.execute("SELECT count(*), sum(articles), sum(passages) FROM built").fetchone()
    print(f"index complete: {totals[0]} files, {totals[1]} articles, {totals[2]} passages")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
