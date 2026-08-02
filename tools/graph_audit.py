# graph_audit.py
"""
Read-only audit of the LightRAG knowledge graph.

Walks every entity in the graph, groups them by normalized name, and reports
problems for a human to review. Writes NOTHING — no merges, no deletes, no
edits. The output is a markdown report you read and act on manually.

What it looks for
-----------------
* duplicates  — same entity under several spellings ("Кирилл Красавчик" vs
                "Kirill Krasavchik", "2500 Скк" vs "2500 скк"). These split
                a person's facts across two nodes so retrieval only ever sees
                half of them.
* junk        — entities that are not facts at all: block headers the
                extractor mistook for things ("04:28 Message",
                "2026-07-25 15:24-15:24"), bare numbers, leftovers from
                pasted articles and prompts.
* orphans     — entities with no relations. Usually extraction noise; they
                inflate the graph without ever being retrieved usefully.
* contradictions — for the chat's main people only, an LLM pass over the
                entity description looking for facts that conflict without a
                clear "which one is current" (jobs, cities, relationships).

Usage
-----
    python tools/graph_audit.py                    # full audit -> graph_audit_report.md
    python tools/graph_audit.py --skip-llm         # structural checks only, no API cost
    python tools/graph_audit.py --people 3         # LLM pass over top-3 people only
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import unicodedata
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import aiohttp

from pathlib import Path

# This tool lives in tools/, one level below the bot's modules. Put the project
# root on sys.path so it runs the same from anywhere: python tools/graph_audit.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from models import BotConfig
from rag_client import RagClient, RagClientError

logger = logging.getLogger("graph_audit")

_REPORT_DEFAULT = "graph_audit_report.md"

# Entity names matching these are almost certainly extraction noise rather than
# real facts about people. Anchored so we only catch whole names, not
# substrings of legitimate ones.
_JUNK_PATTERNS = [
    (r"^\d{4}-\d{2}-\d{2}[\s\d:.-]*$", "block header (date)"),
    (r"^\d{1,2}:\d{2}[\s\d:.-]*(message)?$", "block header (time)"),
    (r"^[\d\s.,:%-]+$", "bare number"),
    (r"^.$", "single character"),
    (r"^(message|сообщение|reply|ответ)$", "chat mechanics"),
]

# Relations below this weight are single-mention noise. Used only to sort the
# report, never to filter anything out.
_STRONG_RELATION_WEIGHT = 5.0


def _normalize(name: str) -> str:
    """
    Collapse a name to a comparison key.

    Unicode NFKC folds the exotic dashes and spaces the LLM emits (U+2011
    non-breaking hyphen showed up in the real graph), then we strip case,
    punctuation and spacing so "Кирилл-Красавчик" and "кирилл красавчик"
    land on the same key.
    """
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"[\s\-_'\"`.,()]+", "", s)
    return s


# Latin<->Cyrillic lookalikes. The extractor sometimes writes a name half in
# each script, which normalization alone cannot catch.
_TRANSLIT = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "b": "в", "h": "н", "k": "к", "m": "м", "t": "т",
})


def _translit_key(name: str) -> str:
    return _normalize(name).translate(_TRANSLIT)


class GraphAuditor:
    """Read-only inspector over the LightRAG graph API."""

    def __init__(self, rag: RagClient, config: BotConfig) -> None:
        self._rag = rag
        self._config = config
        self._nodes: Dict[str, dict] = {}
        self._degree: Dict[str, int] = defaultdict(int)
        self._edges: List[dict] = []

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    async def _get(self, path: str, params: dict, timeout: float) -> dict:
        """GET with query params. RagClient._request only does JSON bodies."""
        token = await self._rag._ensure_token()
        url = f"{self._rag.base_url}{path}?{urllib.parse.urlencode(params)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RagClientError(f"GET {path} -> {resp.status}: {text[:300]}")
                return await resp.json(content_type=None)

    async def fetch_labels(self) -> List[str]:
        data = await self._get("/graph/label/list", {}, timeout=120)
        labels = data if isinstance(data, list) else data.get("labels") or data.get("data") or []
        return [str(x) for x in labels]

    async def fetch_graph(self, max_nodes: int) -> None:
        """
        Pull the whole graph in one shot. `label="*"` is LightRAG's wildcard
        for "everything"; max_nodes caps it server-side.
        """
        data = await self._get(
            "/graphs",
            {"label": "*", "max_depth": 1, "max_nodes": max_nodes},
            timeout=300,
        )
        for node in data.get("nodes", []):
            entity_id = node.get("id") or ""
            if entity_id:
                self._nodes[entity_id] = node
        self._edges = data.get("edges", [])
        for edge in self._edges:
            self._degree[edge.get("source", "")] += 1
            self._degree[edge.get("target", "")] += 1

        if data.get("is_truncated"):
            logger.warning(
                "Graph was truncated at %d nodes — raise --max-nodes for full coverage",
                max_nodes,
            )

    # ------------------------------------------------------------------ #
    # Structural checks (free — no LLM)
    # ------------------------------------------------------------------ #
    def find_duplicates(self, labels: List[str]) -> List[List[str]]:
        """Group labels that normalize to the same key (or transliterate to it)."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for label in labels:
            groups[_normalize(label)].append(label)

        # Second pass over still-unique names, catching mixed-script spellings.
        translit_groups: Dict[str, List[str]] = defaultdict(list)
        for key, names in groups.items():
            if len(names) == 1:
                translit_groups[_translit_key(names[0])].append(names[0])

        found = [sorted(names) for names in groups.values() if len(names) > 1]
        found += [sorted(names) for names in translit_groups.values() if len(names) > 1]
        return sorted(found, key=lambda g: -len(g))

    def find_junk(self, labels: List[str]) -> List[Tuple[str, str]]:
        junk: List[Tuple[str, str]] = []
        for label in labels:
            stripped = label.strip()
            for pattern, reason in _JUNK_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    junk.append((label, reason))
                    break
        return junk

    def find_orphans(self, labels: List[str]) -> List[str]:
        """Entities with no edges. Only meaningful if the graph wasn't truncated."""
        return sorted(l for l in labels if self._degree.get(l, 0) == 0)

    def rank_people(self, limit: int) -> List[str]:
        """
        The chat's main figures, by how connected they are. entity_type comes
        from the extractor and is not always right, so degree is the tiebreaker.
        """
        people = [
            (entity_id, self._degree.get(entity_id, 0))
            for entity_id, node in self._nodes.items()
            if (node.get("properties", {}).get("entity_type") or "").lower() == "person"
        ]
        people.sort(key=lambda x: -x[1])
        return [entity_id for entity_id, _ in people[:limit]]

    def description_of(self, entity_id: str) -> str:
        node = self._nodes.get(entity_id) or {}
        return node.get("properties", {}).get("description") or ""

    # ------------------------------------------------------------------ #
    # LLM check: contradictions inside a person's description
    # ------------------------------------------------------------------ #
    async def find_contradictions(self, entity_id: str) -> Optional[dict]:
        description = self.description_of(entity_id)
        if len(description) < 200:
            return None

        prompt = (
            "Ниже — описание человека из чата, склеенное из сотен упоминаний "
            "за год. Найди в нём ФАКТИЧЕСКИЕ ПРОТИВОРЕЧИЯ: места, где "
            "утверждается одно, а рядом другое, несовместимое с первым "
            "(работа, учёба, город, отношения, возраст).\n\n"
            "Не считай противоречием: шутки, ролевые выдумки, смену мнения о "
            "фильме или игре, разные грани характера.\n\n"
            "Верни JSON: {\"contradictions\": [{\"field\": \"...\", "
            "\"a\": \"...\", \"b\": \"...\", \"note\": \"...\"}]}. "
            "Если противоречий нет — пустой список.\n\n"
            f"ОПИСАНИЕ:\n{description[:6000]}"
        )

        payload = {
            "model": self._config.classifier_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        url = f"{self._config.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.deepseek_api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning("LLM check failed for %s: HTTP %d", entity_id, resp.status)
                        return None
                    data = await resp.json(content_type=None)
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception as e:
            logger.warning("LLM check failed for %s: %s", entity_id, e)
            return None

        items = parsed.get("contradictions") or []
        if not items:
            return None
        return {"entity": entity_id, "items": items}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(
    total_labels: int,
    duplicates: List[List[str]],
    junk: List[Tuple[str, str]],
    orphans: List[str],
    contradictions: List[dict],
    truncated: bool,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Аудит графа знаний")
    add("")
    add("Отчёт только читает граф. Ничего не изменено и не удалено.")
    add("")
    add("## Сводка")
    add("")
    add(f"- Всего сущностей: **{total_labels}**")
    add(f"- Групп дублей: **{len(duplicates)}**")
    add(f"- Мусорных сущностей: **{len(junk)}**")
    add(f"- Сущностей без связей: **{len(orphans) if not truncated else '?'}**")
    add(f"- Людей с противоречиями: **{len(contradictions)}**")
    if truncated:
        add("")
        add("> Сервер отдал только часть графа (лимит на его стороне). "
            "Дубли и мусор проверены по полному списку сущностей, "
            "а поиск сущностей без связей — нет.")
    add("")

    add("## Дубли")
    add("")
    if duplicates:
        add("Одна и та же сущность записана по-разному. Факты о ней разложены "
            "по разным узлам, поэтому поиск находит только половину.")
        add("")
        for group in duplicates:
            add(f"- {' | '.join(group)}")
    else:
        add("Не найдено.")
    add("")

    add("## Мусор")
    add("")
    if junk:
        add("Не факты о людях, а обрывки разметки и вставленного текста, "
            "которые извлекатель принял за сущности.")
        add("")
        for name, reason in junk[:200]:
            add(f"- `{name}` — {reason}")
        if len(junk) > 200:
            add(f"- …ещё {len(junk) - 200}")
    else:
        add("Не найдено.")
    add("")

    add("## Противоречия")
    add("")
    if contradictions:
        for entry in contradictions:
            add(f"### {entry['entity']}")
            add("")
            for item in entry["items"]:
                add(f"- **{item.get('field', '?')}**: "
                    f"«{item.get('a', '')}» против «{item.get('b', '')}»")
                if item.get("note"):
                    add(f"  - {item['note']}")
            add("")
    else:
        add("Не найдено.")
    add("")

    add("## Сущности без связей")
    add("")
    if truncated:
        add("Не проверено: сервер отдал не весь граф.")
    elif orphans:
        add("Ни с чем не связаны — обычно шум извлечения.")
        add("")
        for name in orphans[:100]:
            add(f"- `{name}`")
        if len(orphans) > 100:
            add(f"- …ещё {len(orphans) - 100}")
    else:
        add("Не найдено.")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_audit(args, config: BotConfig) -> int:
    rag = RagClient(
        base_url=config.lightrag_api_url,
        username=config.lightrag_api_user,
        password=config.lightrag_api_password,
    )
    if not await rag.health():
        logger.error("LightRAG unreachable — check LIGHTRAG_API_* settings")
        return 1

    auditor = GraphAuditor(rag, config)

    logger.info("Fetching entity list...")
    labels = await auditor.fetch_labels()
    logger.info("Got %d entities", len(labels))

    logger.info("Fetching graph (up to %d nodes)...", args.max_nodes)
    await auditor.fetch_graph(args.max_nodes)
    logger.info("Got %d nodes, %d relations", len(auditor._nodes), len(auditor._edges))

    truncated = len(auditor._nodes) < len(labels)

    duplicates = auditor.find_duplicates(labels)
    junk = auditor.find_junk(labels)
    # Orphan detection is only trustworthy on a complete graph.
    orphans = [] if truncated else auditor.find_orphans(labels)
    logger.info(
        "Structural: %d duplicate groups, %d junk, %d orphans",
        len(duplicates), len(junk), len(orphans),
    )

    contradictions: List[dict] = []
    if not args.skip_llm:
        people = auditor.rank_people(args.people)
        logger.info("Checking %d people for contradictions: %s", len(people), people)
        results = await asyncio.gather(
            *(auditor.find_contradictions(p) for p in people)
        )
        contradictions = [r for r in results if r]

    report = build_report(
        total_labels=len(labels),
        duplicates=duplicates,
        junk=junk,
        orphans=orphans,
        contradictions=contradictions,
        truncated=truncated,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report written to %s", args.out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the LightRAG knowledge graph."
    )
    parser.add_argument("--out", default=_REPORT_DEFAULT, help="Report file path")
    parser.add_argument("--max-nodes", type=int, default=5000,
                        help="Cap on nodes pulled from the graph")
    parser.add_argument("--people", type=int, default=6,
                        help="How many top people to LLM-check for contradictions")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Structural checks only (no API cost)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        config = load_config()
    except Exception as e:
        logger.error("Could not load config: %s", e)
        return 1

    if not config.lightrag_enabled:
        logger.error("LightRAG is disabled (LIGHTRAG_ENABLED)")
        return 1

    return asyncio.run(run_audit(args, config))


if __name__ == "__main__":
    sys.exit(main())
