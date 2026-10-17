# graph_cleanup.py
"""
Fix the duplicates that string normalization cannot see, and drop junk nodes.

Why this exists
---------------
graph_merge.py groups entities whose names match after folding case and
punctuation. That misses two large classes of duplicate in this graph:

* Transliteration. Early imports ran with `summary_language: English` (since
  switched to Russian), so the extractor rendered proper nouns in Latin script
  some of the time and one person became many nodes: "Максим Тян" (2967
  relations) plus "Maxim Tyan", "Maksim Tyan", "Maksym Tian" and three more,
  together holding ~440 relations the bot never finds when asked about him.
* Nicknames. "Заза" and "Зазик Лайф" are one person; no amount of string
  folding will say so.

Both need a human to assert the identity, so the groups are listed literally
below rather than guessed.

Junk nodes are separate: block headers and media placeholders that leaked out
of the importer and became entities. They carry a handful of relations and no
meaning, so they are deleted rather than merged.

Usage
-----
    python tools/graph_cleanup.py --dry-run     # show what would happen
    python tools/graph_cleanup.py               # apply
    python tools/graph_cleanup.py --junk-only   # junk deletion only (no merges)
    python tools/graph_cleanup.py --skip-junk   # merges only
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import List, Optional, Tuple

import aiohttp

from pathlib import Path

# This tool lives in tools/, one level below the bot's modules. Put the project
# root on sys.path so it runs the same from anywhere: python tools/graph_cleanup.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from graph_audit import _JUNK_PATTERNS as _AUDIT_JUNK_PATTERNS
from graph_merge import _build_rag_client, apply_merge, fetch_labels, inspect_node
from models import BotConfig
from rag_client import RagClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("graph_cleanup")

# One person per entry. The survivor is chosen by relation count at runtime, so
# order here does not matter — only membership does.
_IDENTITY_GROUPS: List[List[str]] = [
    ["Максим Тян", "Maxim Tyan", "Maksim Tyan", "Maksym Tyan",
     "Maksym Tian", "Maksim Tian", "Maxym Tyan"],
    ["Пан Селянин Співак", "Pan Selyanin Spivak", "Селянин Співак", "Пан Селянин"],
    ["Кирилл Красавчик", "Kirill Krasavchik"],
    ["Зазик Лайф", "Zazik Life", "Заза", "заза"],
    ["Tima", "Тима"],
    ["Дип Сик", "Дипсик", "Дип Сикич"],
]

# Extraction artefacts: no meaning, safe to remove outright.
_JUNK: List[str] = [
    "2025-03-04 22:28-22:28",       # a time-block header the LLM read as an entity
    "2025-10-28 23:37-23:39",
    "https://youtu.be/1clWprLC5Ak?si=3MQIBTbnD29L_jTi",
    "https://youtu.be/b51C8AbRDGU",
    "https://youtu.be/oerOLoUy8Tg?si=b4n0agvD45Clzq5j",
    "[Кружок]",                     # media placeholders promoted to entities
    "[Фото]",
    "ᅠ ︎ ︎ ︎ ︎ ᅠ ︎ ︎ ︎ ︎ ᅠ",       # invisible-character node
    # single-character noise from 2025 import
    "X", "Z", "А", "В", "Г", "Н", "С", "Ф", "ф",
    "reply",                        # chat mechanics leaked as entity
    # 2023 import junk
    "2023-08-16", "2023-08-17", "2023-10-04",
    "🤡",
]

# Junk rules are shared with graph_audit on purpose — a single source of truth.
# They catch ONLY extraction noise: block headers, bare numbers, single
# characters, chat mechanics. Measures ("100 км", "183 см", "150 грн",
# "2016 год") are NOT matched — they are real facts and stay in the graph.
def find_junk(labels: List[str]) -> List[Tuple[str, str]]:
    """(label, reason) pairs the audit's rules classify as extraction noise."""
    found: List[Tuple[str, str]] = []
    for label in labels:
        stripped = label.strip()
        for pattern, reason in _AUDIT_JUNK_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                found.append((label, reason))
                break
    return found


async def delete_entity(rag: RagClient, name: str) -> bool:
    for attempt in range(1, 4):
        try:
            token = await rag._ensure_token()
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{rag.base_url}/graph/entity/delete",
                    json={"entity_name": name},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status >= 400:
                        logger.error("  delete %r failed: %s", name, (await resp.text())[:200])
                        return False
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning("  delete %r attempt %d/3: %s", name, attempt, e)
            if attempt == 3:
                return False
            await asyncio.sleep(5 * attempt)
    return False


async def run(args, config: BotConfig) -> int:
    rag = _build_rag_client(config)
    if rag is None or not await rag.health():
        logger.error("LightRAG unavailable — check .env")
        return 1

    present = set(await fetch_labels(rag))
    merged = deleted = failed = 0

    if args.junk_only:
        logger.info("=== Identity merges skipped (--junk-only) ===")
    else:
        logger.info("=== Identity merges ===")
        for group in _IDENTITY_GROUPS:
            members = [m for m in group if m in present]
            if len(members) < 2:
                logger.info("  %s: nothing to merge", group[0])
                continue
            scored = [(m, (await inspect_node(rag, m))[0]) for m in members]
            scored.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            survivor, losers = scored[0][0], [m for m, _ in scored[1:]]
            gained = sum(d for _, d in scored[1:])
            logger.info("  %s (%d) <- %s   [+%d relations]", survivor, scored[0][1],
                        ", ".join(f"{m} ({d})" for m, d in scored[1:]), gained)
            if args.dry_run:
                continue
            if await apply_merge(rag, survivor, losers):
                merged += 1
            else:
                failed += 1

    if not args.skip_junk:
        logger.info("=== Junk deletion ===")
        targets: List[Tuple[str, str]] = [(n, "listed") for n in _JUNK if n in present]
        targets += [(n, r) for n, r in find_junk(sorted(present)) if n not in _JUNK]
        listed = sum(1 for n in _JUNK if n in present)
        logger.info("  %d junk node(s): %d listed, %d matched by pattern",
                    len(targets), listed, len(targets) - listed)
        for i, (name, reason) in enumerate(targets, 1):
            logger.info("  [%d/%d] delete %r (%s)", i, len(targets), name, reason)
            if args.dry_run:
                continue
            if await delete_entity(rag, name):
                deleted += 1
            else:
                failed += 1

    if args.dry_run:
        logger.info("[DRY] nothing changed")
        return 0

    logger.info("Merged %d group(s), deleted %d node(s), %d failure(s). Entities now: %d",
                merged, deleted, failed, len(await fetch_labels(rag)))
    return 0 if failed == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge transliteration/nickname duplicates and delete junk nodes."
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without changing anything")
    parser.add_argument("--skip-junk", action="store_true", help="Only do the identity merges")
    parser.add_argument("--junk-only", action="store_true",
                        help="Only delete junk (skip identity merges)")
    args = parser.parse_args()
    config = load_config()
    sys.exit(asyncio.run(run(args, config)))


if __name__ == "__main__":
    main()
