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
    python tools/graph_cleanup.py --skip-junk   # merges only
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import List, Optional

import aiohttp

from pathlib import Path

# This tool lives in tools/, one level below the bot's modules. Put the project
# root on sys.path so it runs the same from anywhere: python tools/graph_cleanup.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
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
]

# Quantities the extractor mistook for entities: "10 км", "700 грн", "128GB",
# "2024 год", "16:16". They are measurements, not things — the meaning lives in
# whatever event they were attached to, and that survives the delete. Left as
# patterns rather than a fixed list because every import creates a fresh crop.
#
# Deliberately narrow: the label must be JUST a number (optionally with a unit).
# Anything with a real word in it ("100 Km Hike", "2-й батальон 72-ї ОМБр") is a
# possible real entity and is left alone — run --dry-run to review what matched.
_UNIT = (
    r"(?:грн|uah|usd|eur|руб|долл|dollars?|hryvnia|км|km|м|m|см|cm|мм|mm|"
    r"кг|kg|г|g|л|l|мл|ml|тб|tb|гб|gb|мб|mb|мач|mah|hz|гц|fps|k|к|"
    r"ч|h|час(?:а|ов)?|hours?|мин|min|минут\w*|сек|s|градус\w*|degrees?|"
    r"%|процент\w*|тыс\w*|млн|млрд|million|billion)"
)
_NUM = r"[\d]+(?:[.,\s]\d+)*"

_JUNK_PATTERNS = [
    re.compile(r"^[\d\s.,:%\-–—/]+$"),                             # 1600, 15%, 15-0, 2024-2025
    re.compile(r"(?i)^(19|20)\d\d\s*(год|year|г\.?)?$"),           # 2024 год, 2021 Year
    re.compile(r"^\d{1,2}[:.]\d{2}(-\d{1,2}[:.]\d{2})?$"),         # 16:16
    re.compile(r"^\d{4}-\d{2}-\d{2}"),                             # 2024-08-11 23:22-23:22
    re.compile(r"(?i)^[$€₴]?\s*" + _NUM + r"\s*" + _UNIT + r"\.?$"),  # 700 грн, 128GB, 10 км
    re.compile(r"(?i)^[$€₴]\s*" + _NUM + r"$"),                    # $500
]


def find_junk(labels: List[str]) -> List[str]:
    """Labels that are bare quantities/dates rather than entities."""
    return [l for l in labels if any(p.match(l.strip()) for p in _JUNK_PATTERNS)]


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
        targets = [n for n in _JUNK if n in present]
        targets += [n for n in find_junk(sorted(present)) if n not in _JUNK]
        logger.info("  %d junk node(s): %d listed, %d matched by pattern",
                    len(targets), sum(1 for n in _JUNK if n in present),
                    len(targets) - sum(1 for n in _JUNK if n in present))
        for i, name in enumerate(targets, 1):
            logger.info("  [%d/%d] delete %r", i, len(targets), name)
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
    args = parser.parse_args()
    config = load_config()
    sys.exit(asyncio.run(run(args, config)))


if __name__ == "__main__":
    main()
