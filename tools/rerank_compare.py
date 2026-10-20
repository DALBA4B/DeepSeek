# tools/rerank_compare.py
"""
Final comparison: baseline (bot today) vs v1 (word overlap) vs v3 (bot's new
selection, rerank.py).

Scoring lives in rerank.py — the module the bot itself uses, so these numbers
measure exactly what will run in production. Equal char budget for all three
("за ту же цену").

Metrics per question (proxies, honestly labeled):
  junk_kept  — entities with junk names (dates / 1-2 chars) + chunks with
               <8 real words or 15+ media tags. Lower is better.
  cover      — content words of the question found in kept facts. Higher better.
  zeros      — kept facts with no match at all. Lower is better.
  ABSENT?    — no raw match on any content word anywhere in raw output.

Usage: python tools/rerank_compare.py [--dumps dumps]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_client import _SECTION, _json_lines  # noqa: E402
from rerank_exp import select_overlap  # noqa: E402
from rerank import (  # noqa: E402
    chunk_junk,
    common_prefix,
    compute_idf,
    content_words,
    entity_name,
    fact_words,
    is_junk_entity,
    score_fact,
    select_facts,
    stem,
    translit,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="dumps")
    args = ap.parse_args()
    print("question | strategy | junk_kept | cover | zeros_in | absent?")
    print("-" * 70)
    for path in sorted(Path(args.dumps).glob("q*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        q = rec["question"]
        cw = content_words(q)
        m = _SECTION.search(rec.get("raw_context", "") or "")
        if not m:
            continue
        ents = _json_lines(m.group(1))
        chunks = _json_lines(m.group(3))
        facts = []
        for i, e in enumerate(ents):
            facts.append(("entity", i, f"ENTITY {entity_name(e)}: {e.get('description', '')}"))
        for i, c in enumerate(chunks):
            facts.append(("chunk", i, f"CHUNK: {c.get('content', c.get('text', ''))}"))
        budget = len(rec.get("compacted") or "")
        max_chunks = rec.get("max_chunks") or len(chunks)

        # idf with +1 floor: a word present everywhere still distinguishes
        # "match" (score>0) from "no match" (score==0). Without the floor,
        # ubiquitous words score exactly 0 and look like misses.
        idf = compute_idf([t for _, _, t in facts], cw)
        facts4 = [
            ("entity", i, entity_name(ents[i]), t) if k == "entity" else (k, i, None, t)
            for k, i, t in facts
        ]

        def metrics(kept):
            jk = sum(1 for k, i, t in kept
                     if (k == "entity" and is_junk_entity(entity_name(ents[i])))
                     or (k == "chunk" and chunk_junk(t)))
            cov = sum(1 for w in cw if any(
                w in fact_words(t) or
                any(len(f) >= 4 and common_prefix(f, stem(w)) >= 4 for f in fact_words(t))
                or translit(w) in fact_words(t) for _, _, t in kept))
            return jk, f"{cov}/{len(cw)}"

        # baseline: all entities + first max_chunks
        base = [("entity", i, t) for i, (_, _, t) in
                enumerate([f for f in facts if f[0] == "entity"])]
        base += [(k, i, t) for k, i, t in facts
                 if k == "chunk" and i < max_chunks]
        # v1
        _, v1kept = select_overlap(rec.get("raw_context", ""), q, budget,
                                   rec.get("desc_cap") or 0)
        v1 = [(k["kind"], k["idx"],
               next(t for kk, ii, t in facts if kk == k["kind"] and ii == k["idx"]))
              for k in v1kept]
        # v3 — the module the bot ships
        kept4, v3stats = select_facts(facts4, q, budget)
        v3 = [(k, i, t) for k, i, _, t in kept4]
        absent = v3stats["absent"]

        for name, kept in (("baseline", base), ("v1", v1), ("v3", v3)):
            jk, cov = metrics([(k, i, t) for k, i, t in kept])
            zeros = sum(1 for k, i, t in kept if score_fact(t, cw, idf)[0] == 0)
            print(f"{path.name} {q[:28]:28s} | {name:8s} | {jk:2d} | {cov:5s} | {zeros:2d} | "
                  + ("ABSENT" if absent and name == "v3" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
