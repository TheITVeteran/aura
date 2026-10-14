#!/usr/bin/env python3
"""How often her corpus's answer is in the passages she would be given: by words alone, and ordered by her embedder.

For SimpleQA development questions, the careful lookup
(core/knowledge/passages_about.py) is run with and without her embedder's
reranking, and a question counts when every word of its gold answer is in
the passages returned within the first k. Questions are embedded for the
"document" task, passages as documents, by her runtime's own engine
(core/memory/vector_memory_engine.py). When the engine cannot load its model
(another process holds the model lane) it falls back to TF-IDF, and this
refuses rather than report a number from a different space.

Usage:
    measure_passage_recall.py --count 200 --seed 7 --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WITHIN = (1, 3, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    from core.knowledge.local_corpus import LocalCorpusStore
    from core.knowledge.passages_about import passages_about
    from core.memory.vector_memory_engine import EmbeddingEngine
    from tools.run_g09_organ import sample
    from tools.run_knowledge_with_corpus import _words, tasks

    engine = EmbeddingEngine()
    engine.embed_query("a probe", task="document")
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load (the model lane is held elsewhere); nothing measured")

    def rerank(question: str, paragraphs: list[str]) -> list[float]:
        query = np.asarray(engine.embed_query(question, task="document"), dtype=np.float32)
        docs = np.asarray(engine.embed_batch(paragraphs), dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-9)
        docs /= np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9)
        return [float(x) for x in docs @ query]

    def holds(passages: list[tuple[str, str]], gold: str) -> bool:
        text = set(_words(" ".join(f"{t} {p}" for t, p in passages)))
        need = _words(gold)
        return bool(need) and all(w in text for w in need)

    store = LocalCorpusStore()
    chosen = sample(tasks("development"), args.count, args.seed, set())
    counts = {"words": dict.fromkeys(WITHIN, 0), "meaning": dict.fromkeys(WITHIN, 0)}
    started = time.monotonic()
    for task in chosen:
        by_words = passages_about(task["request"], store=store, limit=max(WITHIN))
        for k in WITHIN:
            counts["words"][k] += holds(by_words[:k], task["truth"])
            # Each k as a live lookup carrying k passages would run it: its own pool of four per passage.
            counts["meaning"][k] += holds(passages_about(task["request"], store=store, limit=k, rerank=rerank),
                                          task["truth"])
    report = {"questions": len(chosen), "seed": args.seed, "answer_within_k": counts,
              "seconds": round(time.monotonic() - started, 1)}
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
