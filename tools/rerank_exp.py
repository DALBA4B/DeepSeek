# tools/rerank_exp.py
"""
Offline rerank experiments on recorded retrieval dumps. No server, no API
calls, no cost — runs on the notebook against dumps/*.json produced by
tools/dump_retrieval.py.

Compares, at an EQUAL character budget ("за ту же цену"):
  baseline — what the bot sends today (compacted blob from the dump);
  overlap  — same budget filled with top facts by word overlap with the question.

Output: dumps/report.md with per-question side-by-side blocks for human
judgment. The verdict which selection is better is yours, not the script's:
read both variants and mark which one you would rather give the model.

Usage:
    python tools/rerank_exp.py [--dumps dumps] [--out dumps/report.md]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_client import _SECTION, _json_lines, _shrink_description  # noqa: E402

_WORD = re.compile(r"[a-zа-яё]+", re.I)
_STOP = {
    "что", "это", "как", "кто", "такой", "такая", "чем", "где", "когда",
    "или", "для", "про", "чем", "был", "была", "было", "есть", "уже",
    "the", "and", "who", "what", "чем", "самая", "самый", "расскажи",
}


def query_words(question: str) -> set:
    return {w for w in _WORD.findall(question.lower()) if len(w) >= 3 and w not in _STOP}


def split_facts(raw: str) -> tuple:
    """Return (entities, chunks) as lists of dicts. Empty lists if layout unknown."""
    m = _SECTION.search(raw or "")
    if not m:
        return [], []
    ent_block, _, chunk_block, _ = m.groups()
    return _json_lines(ent_block), _json_lines(chunk_block)


def ent_name(item: dict) -> str:
    # LightRAG names the key "entity" (not entity_name / name)
    return item.get("entity") or item.get("entity_name") or item.get("name") or "?"


def fact_text(kind: str, item: dict) -> str:
    if kind == "entity":
        return f"ENTITY {ent_name(item)}: {item.get('description', '')}"
    return f"CHUNK: {item.get('content', item.get('text', ''))}"


def overlap_score(text: str, qwords: set) -> int:
    low = text.lower()
    return sum(1 for w in qwords if w in low)


def select_overlap(raw: str, question: str, budget: int, desc_cap: int) -> tuple:
    """Fill `budget` chars with top-overlap facts. Returns (text, kept_info)."""
    entities, chunks = split_facts(raw)
    qwords = query_words(question)
    scored = []
    for i, ent in enumerate(entities):
        desc = _shrink_description(ent.get("description", ""), desc_cap) if desc_cap else ent.get("description", "")
        t = f"ENTITY {ent_name(ent)}: {desc}"
        scored.append((overlap_score(t, qwords), 0, i, t))  # entities first on ties
    for i, ch in enumerate(chunks):
        t = f"CHUNK: {ch.get('content', ch.get('text', ''))}"
        scored.append((overlap_score(t, qwords), 1, i, t))
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    out, used, kept = [], 0, []
    for score, kind, i, t in scored:
        if used + len(t) > budget and out:
            continue
        out.append(t)
        kept.append({"score": score, "kind": "entity" if kind == 0 else "chunk", "idx": i})
        used += len(t)
        if used >= budget:
            break
    return "\n".join(out), kept


def short(text: str, n: int = 150) -> str:
    t = " ".join(text.split())
    return t[:n] + ("..." if len(t) > n else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline rerank experiment on dumps.")
    ap.add_argument("--dumps", default="dumps")
    ap.add_argument("--out", default="dumps/report.md")
    args = ap.parse_args()
    dump_dir = Path(args.dumps)
    files = sorted(dump_dir.glob("q*.json"))
    if not files:
        print(f"No dumps in {dump_dir} — run tools/dump_retrieval.py first.")
        return 1
    lines = ["# Rerank experiment: baseline vs word-overlap", ""]
    lines += ["Как читать: у каждого вопроса два списка фактов — что берет бот",
              "сегодня (baseline) и что взял бы отбор (overlap). Бюджет одинаковый.",
              "Смотришь только имена/первые строки и блок ЗАМЕНА.",
              "Вердикт — одна строка на вопрос.", ""]
    for path in files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        q = rec["question"]
        baseline = rec.get("compacted") or ""
        budget = len(baseline)
        entities, chunks = split_facts(rec.get("raw_context", ""))
        _, kept = select_overlap(
            rec.get("raw_context", ""), q, budget, rec.get("desc_cap") or 0
        )
        kept_set = {(k["kind"], k["idx"]) for k in kept}
        max_chunks = rec.get("max_chunks") or len(chunks)
        base_set = {("entity", i) for i in range(len(entities))}
        base_set |= {("chunk", i) for i in range(min(max_chunks, len(chunks)))}
        ent_names = [ent_name(e) for e in entities]
        chunk_heads = [short(c.get("content", c.get("text", ""))) for c in chunks]

        def fmt(kind, i):
            if kind == "entity":
                name = ent_names[i] if i < len(ent_names) else "?"
                return f"E{i:02d} {short(name, 60)}"
            head = chunk_heads[i] if i < len(chunk_heads) else ""
            return f"C{i:02d} {head}"

        lines += [
            f"## {path.name}: {q}",
            f"сырого: {rec.get('raw_chars')}, бюджет: {budget}, "
            f"сущностей: {len(entities)}, чанков всего: {len(chunks)}",
            "",
            f"### Baseline (бот сегодня): {len(base_set)} фактов",
            "",
        ]
        for kind, i in sorted(base_set):
            lines.append(f"- {fmt(kind, i)}")
        lines += ["", f"### Overlap (отбор): {len(kept_set)} фактов", ""]
        for k in kept:
            lines.append(f"- [score {k['score']}] {fmt(k['kind'], k['idx'])}")
        only_base = sorted(base_set - kept_set)
        only_over = sorted(kept_set - base_set)
        lines += ["", "### Замена (что поменялось)", ""]
        if not only_base and not only_over:
            lines.append("наборы совпали — отбор ничего не меняет")
        else:
            for kind, i in only_base:
                lines.append(f"- ВЫКИНУЛ БЫ: {fmt(kind, i)}")
            for kind, i in only_over:
                lines.append(f"- ВЗЯЛ БЫ: {fmt(kind, i)}")
        lines += ["", f"VERDICT {path.name} (допиши: baseline / overlap / tie / оба мусор):",
                  "", "---", ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(files)} questions compared -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
