# rerank.py
"""
Client-side rerank (v3): decide which retrieved facts reach the prompt.

Why this exists
---------------
LightRAG answers /query with a wide raw context; the bot trims it in
compact_context() (rag_client.py) by capping entity descriptions and keeping
the first N excerpts in SERVER order. Server order is blind to the question:
a chunk made of photo tags can take a slot from a fact that answers it. The
server-side rerank provider is not configured (rerank_binding='null') and
would need an external paid service, so the ordering is done here, for free.

How v3 scores ("за ту же цену": same character budget as the plain trim)
------------------------------------------------------------------------
* query words: whole Cyrillic alphabet INCLUDING Ukrainian і/ї/є/ґ — the old
  [а-яё] tokenizer split "Співак" into "сп"+"вак" and the question stopped
  matching its own entity; stop-list drops function words;
* every fact (entity text / chat excerpt) gets sum over the query's content
  words of IDF weights: raw match 1.0, stem/transliteration bridge 0.7 (rare
  words weigh more — "максим" is everywhere, its IDF is near the 1.0 floor);
* junk demotion x0.25: entities named like a date or 1-2 characters, chunks
  that are nearly all media tags or have <8 real words. Measures ("100 км"),
  years and money are NOT junk — junk rules live in one place, is_junk_entity
  and chunk_junk;
* absence gate: no RAW match of any query word anywhere means the graph
  probably has nothing — stem/translit lookalikes (Тимур vs Тима) are
  similarity, not knowledge, and must not count as "facts found".

Validated offline on 48 recorded retrievals (dumps/ dumps2/ dumps3/):
coverage never below the plain trim, junk and zero-score facts mostly lower,
see plans/RERANK_V3_PLAN.md.

Pure functions, no I/O. tools/rerank_compare.py imports this module, so the
offline numbers and the bot score facts with literally the same code.
"""

import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Query/fact tokenizer. Ukrainian letters matter: the extractor writes
# "Пан Селянин Співак" and queries come back mixed-script.
_WORD = re.compile(r"[a-zа-яёіїєґ]+", re.I)

# Function words carry no retrieval signal. Union of the stop-lists used by
# the offline experiments (tools/rerank_exp.py, tools/rerank_compare.py).
_STOP = {
    "что", "это", "как", "кто", "такой", "такая", "какая", "какой",
    "какое", "какие", "чем", "где", "когда", "почему", "зачем", "сколько",
    "или", "для", "про", "был", "была", "было", "есть", "уже", "самый",
    "самая", "самое", "лучший", "лучшая", "лучшее", "лучше", "покажи",
    "найди", "расскажи", "the", "and", "who", "what",
}

# Latin<->Cyrillic: RU->EN transliteration so a Russian query word can hit an
# English description in the graph ("тима" -> "tima").
_RU_EN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ь": "", "ы": "y", "ъ": "", "э": "e", "ю": "yu", "я": "ya",
}

_MEDIA_TAGS = ("[фото]", "[видео]", "[стикер]", "[кружок]", "[файл]", "[голосовое]")
_DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NON_WORD = re.compile(r"[\s\-_'\"`.,()]+")

# Junk facts keep their chance: demoted 4x, they enter the prompt only when
# better-scoring facts have not filled the budget.
JUNK_PENALTY = 0.25

# Non-raw matches (stem / translit) count less than a literal hit.
BRIDGE_WEIGHT = 0.7

# Minimum prefix length for two words to count as a stem match.
_MIN_PREFIX = 4

# Appended to the facts block when the absence gate fires: lets the model
# answer "не знаю" instead of inventing. Deliberately cautious wording.
ABSENT_NOTE = "\nПрямых упоминаний в базе знаний не найдено."


def norm(w: str) -> str:
    return unicodedata.normalize("NFKC", w).casefold().replace("ё", "е")


def stem(w: str) -> str:
    """Crude Russian stem: drop one trailing char for words longer than 4."""
    w = norm(w)
    return w[:-1] if len(w) > 4 else w


def translit(w: str) -> str:
    return "".join(_RU_EN.get(c, c) for c in norm(w))


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def fact_words(text: str) -> set:
    return set(_WORD.findall(norm(text)))


def content_words(question: str) -> set:
    """Meaningful words of the question — what the ranking scores against."""
    return {w for w in _WORD.findall(norm(question)) if len(w) >= 3 and w not in _STOP}


def entity_name(item: dict) -> str:
    # LightRAG names the key "entity" (not entity_name / name)
    return item.get("entity") or item.get("entity_name") or item.get("name") or "?"


def is_junk_entity(name: str) -> bool:
    """Date-shaped or 1-2 character names: block headers and scraps, not facts."""
    n = _NON_WORD.sub("", name.casefold())
    return bool(_DATE_NAME.match(name.strip())) or len(n) <= 2


def chunk_junk(text: str) -> bool:
    """Photo dumps and near-empty excerpts: little text, many media tags."""
    low = text.lower()
    tags = sum(low.count(t) for t in _MEDIA_TAGS)
    real = len([w for w in text.split() if w.lower() not in _MEDIA_TAGS])
    return real < 8 or tags >= 15


def compute_idf(texts: List[str], cwords: set) -> Dict[str, float]:
    """
    IDF over the query words, document frequency counted on the union of
    raw + stem + translit matches so "максим" (in every fact) scores near 0.

    The +1 floor keeps a match distinguishable from a miss: without it a word
    present everywhere scores exactly 0.0 and looks like "not found".
    """
    fws = [fact_words(t) for t in texts]
    n = len(fws)
    idf = {}
    for w in cwords:
        s, tr = stem(w), translit(w)
        hits = sum(
            1 for fw in fws
            if w in fw
            or any(len(f) >= _MIN_PREFIX and common_prefix(f, s) >= _MIN_PREFIX for f in fw)
            or tr in fw
        )
        idf[w] = 1.0 + math.log((n + 1) / (hits + 1))
    return idf


def score_fact(text: str, cwords: set, idf: Dict[str, float]) -> Tuple[float, List[str]]:
    """Return (score, match kinds) for one fact text against the question."""
    fw = fact_words(text)
    sc, parts = 0.0, []
    for w in cwords:
        s, tr = stem(w), translit(w)
        if w in fw:
            sc += idf[w]
            parts.append(f"{w}:raw")
        elif any(len(f) >= _MIN_PREFIX and common_prefix(f, s) >= _MIN_PREFIX for f in fw):
            sc += idf[w] * BRIDGE_WEIGHT
            parts.append(f"{w}:stem")
        elif tr in fw:
            sc += idf[w] * BRIDGE_WEIGHT
            parts.append(f"{w}:tr")
    return sc, parts


def is_absent(texts: List[str], cwords: set) -> bool:
    """No RAW match of any query word anywhere in the raw output."""
    fws = [fact_words(t) for t in texts]
    return not any(w in fw for w in cwords for fw in fws)


def select_facts(
    facts: List[Tuple[str, int, Optional[str], str]],
    question: str,
    budget: int,
) -> Tuple[List[Tuple[str, int, Optional[str], str]], dict]:
    """
    Pick facts for the prompt within `budget` characters, best score first.

    Args:
        facts:    [(kind, idx, subject, text)] in server order.
                  kind: 'entity' | 'chunk'; subject: entity name for the junk
                  check (None for chunks); text: "ENTITY <name>: <desc>" /
                  "CHUNK: <excerpt>".
        question: The user's natural-language query.
        budget:   Character budget (the length of the plain trim's output —
                  the selection stays "за ту же цену").

    Returns:
        (kept facts in prompt order, stats). Zero-score facts are NOT thrown
        away: they fill the remaining budget last, in server order — insurance
        against paraphrases the matcher cannot see.
    """
    cwords = content_words(question)
    texts = [t for _, _, _, t in facts]
    idf = compute_idf(texts, cwords)

    scored = []
    for pos, (kind, idx, subject, text) in enumerate(facts):
        sc, _ = score_fact(text, cwords, idf)
        if kind == "chunk" and chunk_junk(text):
            sc *= JUNK_PENALTY
        if kind == "entity" and subject is not None and is_junk_entity(subject):
            sc *= JUNK_PENALTY
        scored.append((sc, pos, kind, idx, subject, text))

    # Most relevant first; ties keep server order (stable by position).
    scored.sort(key=lambda s: (-s[0], s[1]))

    kept: List[Tuple[str, int, Optional[str], str]] = []
    used = 0
    for _, _, kind, idx, subject, text in scored:
        if used + len(text) > budget and kept:
            continue
        kept.append((kind, idx, subject, text))
        used += len(text)
        if used >= budget:
            break

    absent = is_absent(texts, cwords)
    stats = {
        "budget": budget,
        "used": used,
        "facts": len(facts),
        "kept": len(kept),
        "kept_entities": sum(1 for k, _, _, _ in kept if k == "entity"),
        "kept_chunks": sum(1 for k, _, _, _ in kept if k == "chunk"),
        "absent": absent,
    }
    return kept, stats
