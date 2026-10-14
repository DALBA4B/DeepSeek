# tools/rerank_compare.py
"""
Final comparison: baseline (bot today) vs v1 (word overlap) vs v3 prototype.

v3 prototype = stop-list + IDF + bridges + junk demotion + absence gate.
Equal char budget for all three ("за ту же цену").

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
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_client import _SECTION, _json_lines  # noqa: E402
from rerank_exp import query_words, ent_name, select_overlap  # noqa: E402
from rerank_v2check import stem, translit, common_prefix, fact_words  # noqa: E402

_EXTRA_STOP = {
    "какая", "какой", "какое", "какие", "что", "кто", "как", "где",
    "когда", "почему", "зачем", "сколько", "самый", "самая", "самое",
    "лучший", "лучшая", "лучшее", "лучше", "покажи", "найди", "такой",
}
_TAGS = ("[фото]", "[видео]", "[стикер]", "[кружок]", "[файл]", "[голосовое]")
_JUNK_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def content_words(question: str) -> set:
    return {w for w in query_words(question) if w not in _EXTRA_STOP}


def is_junk_entity(name: str) -> bool:
    n = re.sub(r"[\s\-_'\"`.,()]+", "", name.casefold())
    return bool(_JUNK_NAME.match(name.strip())) or len(n) <= 2


def chunk_junk(text: str) -> bool:
    low = text.lower()
    tags = sum(low.count(t) for t in _TAGS)
    real = len([w for w in text.split() if w.lower() not in _TAGS])
    return real < 8 or tags >= 15


def v3_score(text: str, cwords: set, idf: dict) -> tuple:
    fw = fact_words(text)
    sc, parts = 0.0, []
    for w in cwords:
        s, tr = stem(w), translit(w)
        if w in fw:
            sc += idf[w]; parts.append("raw")
        elif any(len(f) >= 4 and common_prefix(f, s) >= 4 for f in fw):
            sc += idf[w] * 0.7; parts.append("stem")
        elif tr in fw:
            sc += idf[w] * 0.7; parts.append("tr")
    return sc, parts


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
            facts.append(("entity", i, f"ENTITY {ent_name(e)}: {e.get('description', '')}"))
        for i, c in enumerate(chunks):
            facts.append(("chunk", i, f"CHUNK: {c.get('content', c.get('text', ''))}"))
        budget = len(rec.get("compacted") or "")
        max_chunks = rec.get("max_chunks") or len(chunks)

        # idf over content words
        # idf with +1 floor: a word present everywhere still distinguishes
        # "match" (score>0) from "no match" (score==0). Without the floor,
        # ubiquitous words score exactly 0 and look like misses.
        fws = [fact_words(t) for _, _, t in facts]
        N = len(facts)
        idf = {}
        for w in cw:
            s, tr = stem(w), translit(w)
            c = sum(1 for fw in fws
                    if w in fw or any(len(f) >= 4 and common_prefix(f, s) >= 4 for f in fw)
                    or tr in fw)
            idf[w] = 1.0 + math.log((N + 1) / (c + 1))

        def metrics(kept):
            jk = sum(1 for k, i, t in kept
                     if (k == "entity" and is_junk_entity(ent_name(ents[i])))
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
        # v3
        scored = []
        for k, i, t in facts:
            sc, parts = v3_score(t, cw, idf)
            if k == "chunk" and chunk_junk(t):
                sc *= 0.25
            if k == "entity" and is_junk_entity(ent_name(ents[i])):
                sc *= 0.25
            scored.append((sc, k, i, t))
        # Absence gate on RAW matches only: stemming/translit find lookalikes
        # (Тиму for Тимур) which must not count as "knowledge exists".
        absent = not any(w in fw for w in cw for fw in fws)
        scored.sort(key=lambda s: (-s[0], facts.index((s[1], s[2], s[3]))))
        v3, used = [], 0
        for sc, k, i, t in scored:
            if used + len(t) > budget and v3:
                continue
            v3.append((k, i, t))
            used += len(t)
            if used >= budget:
                break
        for name, kept in (("baseline", base), ("v1", v1), ("v3", v3)):
            jk, cov = metrics([(k, i, t) for k, i, t in kept])
            zeros = sum(1 for k, i, t in kept if v3_score(t, cw, idf)[0] == 0)
            print(f"{path.name} {q[:28]:28s} | {name:8s} | {jk:2d} | {cov:5s} | {zeros:2d} | "
                  + ("ABSENT" if absent and name == "v3" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
