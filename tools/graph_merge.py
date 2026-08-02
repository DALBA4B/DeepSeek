# graph_merge.py
"""
Merge duplicate entities in the LightRAG knowledge graph.

Why this exists
---------------
The extractor compares entity names literally, so "Тик Ток", "тик ток" and
"ТикТок" become three separate nodes and every fact about TikTok is split
between them — a query finds one third of what the graph knows. graph_audit.py
reports these groups; this script applies the fix through LightRAG's
/graph/entities/merge endpoint (relationships move to the surviving node, the
duplicates are deleted).

Choosing the survivor
---------------------
The variant with the most relationships wins: it already carries the most facts,
so merging into it moves the least data and keeps the node id that retrieval has
been hitting. Ties go to the longest name, which is usually the properly spelled
one ("Зазик Лайф" over "Заза").

Safety
------
* --dry-run prints the plan without touching anything (default: review first).
* Merging is IRREVERSIBLE — LightRAG deletes the source nodes.
* _NEVER_MERGE holds pairs that only *look* like duplicates. Real example: the
  bot is "Дип Сик", while "DeepSeek"/"Deep Seek" is the AI model the chat talks
  about and tries to jailbreak. Same name, different things.

Usage
-----
    python tools/graph_merge.py --dry-run          # show the plan
    python tools/graph_merge.py                    # apply it
    python tools/graph_merge.py --only "Заза"      # just the groups matching a name
"""

import argparse
import asyncio
import logging
import re
import sys
import unicodedata
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import aiohttp

from pathlib import Path

# This tool lives in tools/, one level below the bot's modules. Put the project
# root on sys.path so it runs the same from anywhere: python tools/graph_merge.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from models import BotConfig
from rag_client import RagClient, RagClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("graph_merge")

# Entity names that normalize alike but are genuinely different things. Listed
# as sets: if a duplicate group contains members from two different sets here,
# the group is skipped and reported for manual review.
#
# Verified against the node descriptions 2026-08-01: the misspelled Latin forms
# describe a chat participant ("owns an iPad Pro 2018", "reacts with sarcasm to
# stories about the village"), while DeepSeek/Deep Seek describe the Chinese
# model ("surpassed ChatGPT in benchmarks"). So the typos belong with the bot.
_NEVER_MERGE = [
    {"дип сик", "дипсик", "дип сикич",          # the bot
     "deep sick", "dip syk", "deep sik"},       # ...as misheard by the extractor
    {"deepseek", "deep seek"},                  # the AI model discussed in chat
]

_CONCURRENCY = 4

# Content words shorter than this carry no signal ("the", "и", "a").
_MIN_WORD = 4


def _content_words(text: str, exclude: str = "") -> set:
    """Lowercased words of 4+ chars, minus those already in the entity name."""
    stop = {w for w in re.findall(r"\w+", exclude.casefold()) if len(w) >= _MIN_WORD}
    return {
        w for w in re.findall(r"\w+", text.casefold())
        if len(w) >= _MIN_WORD and w not in stop
    } - {"sep"}


def _descriptions_conflict(name: str, descriptions: List[str]) -> bool:
    """
    True when two same-spelled entities look like genuinely different things.

    Real case from this graph: "Zoom Feature" (screen zoom, broken then fixed)
    and "Zoom feature" (camera zoom, 500x, Chinese phone makers) normalize
    alike but are unrelated. Merging them fuses two facts into one node and the
    bot then answers confidently wrong — worse than leaving a duplicate, which
    only splits recall.

    Words from the entity name are excluded: both descriptions of "Zoom Feature"
    naturally contain "zoom", and that agreement means nothing.
    """
    sets = [_content_words(d, exclude=name) for d in descriptions if d and d.strip()]
    sets = [s for s in sets if len(s) >= 5]  # too short to judge
    if len(sets) < 2:
        return False
    # Any pair sharing no content word at all is suspicious.
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if not (sets[i] & sets[j]):
                return True
    return False


def _normalize(name: str) -> str:
    """Fold case, spacing and punctuation so spelling variants collide."""
    s = unicodedata.normalize("NFKC", name).casefold()
    return re.sub(r"[\s\-_'\"`.,()]+", "", s)


def _protected_group(members: List[str]) -> bool:
    """True when a group mixes entities from two different _NEVER_MERGE sets."""
    hits = set()
    for member in members:
        key = member.casefold().strip()
        for i, group in enumerate(_NEVER_MERGE):
            if key in group:
                hits.add(i)
    return len(hits) > 1


async def _get(rag: RagClient, path: str, params: dict, timeout: float) -> dict:
    token = await rag._ensure_token()
    url = f"{rag.base_url}{path}?{urllib.parse.urlencode(params)}"
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                raise RagClientError(f"GET {path} -> {resp.status}: {(await resp.text())[:200]}")
            return await resp.json(content_type=None)


async def fetch_labels(rag: RagClient) -> List[str]:
    data = await _get(rag, "/graph/label/list", {}, timeout=180)
    labels = data if isinstance(data, list) else data.get("labels") or data.get("data") or []
    return [str(x) for x in labels]


async def inspect_node(rag: RagClient, label: str) -> Tuple[int, str]:
    """Return (relation count, description) for one entity in a single request."""
    for attempt in range(3):
        try:
            data = await _get(
                rag, "/graphs", {"label": label, "max_depth": 1, "max_nodes": 1000}, timeout=120
            )
            desc = ""
            for node in data.get("nodes", []):
                if str(node.get("id")) == label:
                    desc = str((node.get("properties") or {}).get("description") or "")
                    break
            return len(data.get("edges", [])), desc
        except RagClientError:
            return 0, ""
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            if attempt == 2:
                return 0, ""
            await asyncio.sleep(5 * (attempt + 1))
    return 0, ""


def group_duplicates(labels: List[str]) -> List[List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for label in labels:
        groups[_normalize(label)].append(label)
    return [sorted(v) for v in groups.values() if len(v) > 1]


async def plan_merges(
    rag: RagClient, groups: List[List[str]], force: bool = False
) -> Tuple[List[Tuple[str, List[str]]], List[List[str]]]:
    """Decide the survivor for each group. Returns (merges, skipped)."""
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def inspect(members: List[str]) -> List[Tuple[str, int, str]]:
        async def one(m: str) -> Tuple[str, int, str]:
            async with semaphore:
                degree, desc = await inspect_node(rag, m)
                return m, degree, desc
        return list(await asyncio.gather(*(one(m) for m in members)))

    merges: List[Tuple[str, List[str]]] = []
    skipped: List[List[str]] = []
    for group in groups:
        if _protected_group(group):
            skipped.append(group)
            continue
        scored = await inspect(group)
        if not force and _descriptions_conflict(group[0], [d for _, _, d in scored]):
            skipped.append(group)
            continue
        # Most relationships wins; longest name breaks ties.
        scored.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        survivor = scored[0][0]
        losers = [m for m, _, _ in scored[1:]]
        if losers:
            merges.append((survivor, losers))
            logger.info(
                "  %s <- %s", survivor,
                ", ".join(f"{m} ({d})" for m, d, _ in scored[1:]),
            )
    return merges, skipped


async def apply_merge(rag: RagClient, survivor: str, losers: List[str]) -> bool:
    """
    Merge one group, retrying transient failures.

    A run takes tens of minutes and Railway occasionally drops a connection; an
    unhandled ClientError used to abort the whole script and leave the graph
    half-merged.
    """
    payload = {"entities_to_change": losers, "entity_to_change_into": survivor}
    for attempt in range(1, 4):
        try:
            token = await rag._ensure_token()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{rag.base_url}/graph/entities/merge",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status >= 400:
                        logger.error("  merge %s failed: %s", survivor, (await resp.text())[:200])
                        return False
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning("  merge %s attempt %d/3: %s", survivor, attempt, e)
            if attempt == 3:
                return False
            await asyncio.sleep(5 * attempt)
    return False


def _build_rag_client(config: BotConfig) -> Optional[RagClient]:
    if not config.lightrag_enabled or not config.lightrag_api_url:
        return None
    return RagClient(
        base_url=config.lightrag_api_url,
        username=config.lightrag_api_user,
        password=config.lightrag_api_password,
    )


async def run(args, config: BotConfig) -> int:
    rag = _build_rag_client(config)
    if rag is None or not await rag.health():
        logger.error("LightRAG unavailable — check .env")
        return 1

    labels = await fetch_labels(rag)
    groups = group_duplicates(labels)
    if args.only:
        needle = args.only.casefold()
        groups = [g for g in groups if any(needle in m.casefold() for m in g)]
    logger.info("%d entities, %d duplicate group(s) to handle", len(labels), len(groups))
    if not groups:
        return 0

    logger.info("Planning (counting relations per variant)...")
    merges, skipped = await plan_merges(rag, groups, force=args.force)

    for group in skipped:
        logger.warning("SKIPPED (different things, merge by hand if wrong): %s", " | ".join(group))

    if args.dry_run:
        logger.info("[DRY] would merge %d group(s) — nothing changed", len(merges))
        return 0

    ok = 0
    for survivor, losers in merges:
        if await apply_merge(rag, survivor, losers):
            ok += 1
    logger.info("Merged %d/%d group(s). Entities now: %d",
                ok, len(merges), len(await fetch_labels(rag)))
    return 0 if ok == len(merges) else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge duplicate entities in the LightRAG graph.")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without merging")
    parser.add_argument("--only", help="Only groups containing this substring")
    parser.add_argument(
        "--force", action="store_true",
        help="Merge even when descriptions look unrelated (skip the conflict guard)",
    )
    args = parser.parse_args()
    config = load_config()
    sys.exit(asyncio.run(run(args, config)))


if __name__ == "__main__":
    main()
