# tools/rerank_v2check.py
"""
Self-check: measure whether v2 scoring ideas actually beat v1 overlap.

v2 = v1 word overlap PLUS:
  - norm: casefold, NFKC, yo->e, crude Russian stemming (drop 1 trailing char
    for words longer than 4) so "рома" hits "ромка"/"ромы"/"ромой";
  - translit bridge: query words also transliterated RU->EN so Russian queries
    hit English descriptions ("бывшая" won't (concept), but "тима" hits "tima");
  - photo demotion: chunks that are >40% [media tags] get pushed to the end.

Prints per-question metrics + recovered/lost fact lists. Read-only, offline.
Usage: python tools/rerank_v2check.py [--dumps dumps]
"""

import argparse
import io
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_client import _SECTION, _json_lines  # noqa: E402
from rerank_exp import query_words, ent_name  # noqa: E402

_WORD = re.compile(r"[a-zа-яё]+", re.I)
_TAGS = ("[фото]", "[видео]", "[стикер]", "[кружок]", "[файл]", "[голосовое]")

_RU_EN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ь": "", "ы": "y", "ъ": "", "э": "e", "ю": "yu", "я": "ya",
}


def norm(w: str) -> str:
    return unicodedata.normalize("NFKC", w).casefold().replace("ё", "е")


def stem(w: str) -> str:
    w = norm(w)
    return w[:-1] if len(w) > 4 else w


def translit(w: str) -> str:
    return "".join(_RU_EN.get(c, c) for c in norm(w))


def fact_words(text: str) -> set:
    return set(_WORD.findall(norm(text)))


def photo_frac(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    low = text.lower()
    hits = sum(low.count(t) for t in _TAGS)
    return hits / len(toks)


def v1_score(text: str, qwords: set) -> int:
    low = text.lower()
    return sum(1 for w in qwords if w in low)


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def v2_score(text: str, qwords: set, qstems: set, qtrans: set) -> tuple:
    """Return (score, why). why in {raw, stem, translit, none}."""
    fwords = fact_words(text)
    if qwords & fwords:
        return len(qwords & fwords), "raw"
    stems_hit = {s for s in qstems
                 if any(len(f) >= 4 and common_prefix(f, s) >= 4 for f in fwords)}
    if stems_hit:
        return len(stems_hit), "stem"
    if qtrans & fwords:
        return len(qtrans & fwords), "translit"
    return 0, "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="dumps")
    args = ap.parse_args()
    for path in sorted(Path(args.dumps).glob("q*.json")):
        import json
        rec = json.loads(path.read_text(encoding="utf-8"))
        q = rec["question"]
        qwords = query_words(q)
        qstems = {stem(w) for w in qwords}
        qtrans = {translit(w) for w in qwords if any(ord(c) > 127 for c in w)}
        m = _SECTION.search(rec.get("raw_context", "") or "")
        if not m:
            print(f"{path.name}: layout unknown")
            continue
        ents = _json_lines(m.group(1))
        chunks = _json_lines(m.group(3))
        facts = []
        for i, e in enumerate(ents):
            t = f"ENTITY {ent_name(e)}: {e.get('description', '')}"
            facts.append(("entity", i, t))
        for i, c in enumerate(chunks):
            t = f"CHUNK: {c.get('content', c.get('text', ''))}"
            facts.append(("chunk", i, t))
        rec_list = []
        for kind, i, t in facts:
            s1 = v1_score(t, qwords)
            s2, why = v2_score(t, qwords, qstems, qtrans)
            ph = photo_frac(t) if kind == "chunk" else 0.0
            rec_list.append({"kind": kind, "idx": i, "v1": s1, "v2": s2,
                             "why": why, "photo": round(ph, 2),
                             "head": " ".join(t.split())[:120]})
        recovered = [r for r in rec_list if r["v1"] == 0 and r["v2"] > 0]
        # IDF-weighted score: rare query words weigh more. df counted on the
        # union of raw+stem+translit matches so "максим" (everywhere) ~= 0.
        import math
        N = len(facts)
        df = {}
        for r in rec_list:
            kind, i, t = r["kind"], r["idx"], None
        # rebuild match sets per query word
        fword_list = []
        for kind, i, t in facts:
            fword_list.append(fact_words(t))
        idf = {}
        for w in qwords:
            s, tr = stem(w), translit(w)
            c = sum(1 for fw in fword_list
                    if w in fw or any(len(f) >= 4 and common_prefix(f, s) >= 4 for f in fw)
                    or tr in fw)
            idf[w] = math.log((N + 1) / (c + 1))
        ranked = []
        for (kind, i, t), fw in zip(facts, fword_list):
            sc, parts = 0.0, []
            for w in qwords:
                s, tr = stem(w), translit(w)
                if w in fw:
                    sc += idf[w]; parts.append(f"{w}:raw")
                elif any(len(f) >= 4 and common_prefix(f, s) >= 4 for f in fw):
                    sc += idf[w]; parts.append(f"{w}:stem")
                elif tr in fw:
                    sc += idf[w]; parts.append(f"{w}:tr")
            ph = photo_frac(t) if kind == "chunk" else 0.0
            if ph > 0.4:
                sc *= 0.25
                parts.append("PHOTOx0.25")
            ranked.append((round(sc, 2), kind, i, "+".join(parts),
                           " ".join(t.split())[:100]))
        ranked.sort(key=lambda r: -r[0])
        print(f"===== {path.name}: {q} =====")
        print(f"facts={len(facts)} idf={ {w: round(v, 2) for w, v in idf.items()} }")
        print(f"--- top-12 by IDF score (negatives at the end):")
        for sc, kind, i, parts, head in ranked[:12]:
            print(f"  {sc:5.2f} {kind}{i} [{parts}]: {head}")
        nz = [r for r in ranked if r[0] == 0]
        print(f"--- zero-score facts: {len(nz)}/{len(facts)} (absence signal)" if nz else "--- no zero-score facts", "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
