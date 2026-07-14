# import_history.py
"""
One-off importer: load an exported Telegram chat history into LightRAG.

Why this exists
---------------
The nightly ingest (`rag_ingestor.py`) only feeds LightRAG the *last day* of
chat. To give the bot deep memory we also want to backfill the whole history
that already happened before it started running. This script does that from a
Telegram Desktop JSON export, reusing the exact same grouping the nightly
ingest uses, so imported history and live history look identical to LightRAG.

Design (matches the discussion / PHASE plans)
---------------------------------------------
* One document per CALENDAR DAY. A year in one request would time out / OOM
  on the LightRAG side; per-day keeps each insert small and gives every
  document an honest dated `file_source` (telegram_import_YYYY-MM-DD).
* Within a day, messages are grouped into time-coherent blocks by
  RagIngestor.group_into_blocks() — the "chunks" of a conversation (a new
  block starts on a long pause or a size cap). Each block header carries the
  full date + time range, e.g. [2026-03-06 21:40-21:52], so LightRAG can
  place events on the calendar and follow their order.
* Chronological order (old -> new): LightRAG builds its graph incrementally,
  so feeding time in order helps "happened before/after" relations.
* Sequential inserts with a small retry/backoff — a single flaky night must
  not abort the whole backfill.
* Resumable: every successfully inserted day is recorded in a progress file
  (default import_progress.json). Re-running skips days already done, so you
  can stop/restart freely and go month by month.

Getting the export
------------------
Telegram Desktop -> chat -> ⋮ -> Export chat history -> format **JSON**
(not HTML). You get a result.json with a top-level "messages" array.

Usage
-----
    # dry run: parse + group + show what WOULD be inserted, no network
    python import_history.py --file result.json --dry-run

    # import everything (resumes automatically if interrupted)
    python import_history.py --file result.json

    # one month at a time (recommended: verify /profile between months)
    python import_history.py --file result.json --since 2026-03-01 --until 2026-03-31

    # re-import days already done (ignore progress file)
    python import_history.py --file result.json --force

Notes
-----
* Reads LightRAG connection from the same .env / config.py the bot uses.
* Does NOT touch Firebase or the nightly ingest cursor — it only inserts
  documents into LightRAG. Safe to run alongside a live bot.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional

from config import load_config
from models import BotConfig, ChatMessage
from rag_client import RagClient, RagClientError
from rag_ingestor import RagIngestor
from utils import to_aware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("import_history")

_PROGRESS_FILE_DEFAULT = "import_progress.json"
_INSERT_MAX_ATTEMPTS = 3
_INSERT_RETRY_BASE_DELAY = 2.0  # seconds; backfill is not latency-sensitive


# ---------------------------------------------------------------------------
# Telegram JSON export parsing
# ---------------------------------------------------------------------------
def _flatten_text(text_field) -> str:
    """
    Telegram's `text` is either a plain string or a list mixing plain strings
    and {"type": ..., "text": ...} entity dicts (links, mentions, bold, ...).
    Flatten both shapes to a single plain string.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: List[str] = []
        for part in text_field:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def _parse_from_id(raw) -> int:
    """
    Telegram `from_id` looks like "user123456" or "channel123". Extract the
    numeric part; fall back to 0 when it is missing/unparseable.
    """
    if raw is None:
        return 0
    s = str(raw)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def parse_export(path: str) -> List[ChatMessage]:
    """
    Parse a Telegram Desktop JSON export into ChatMessage objects.

    * Skips service messages (joins/pins/calls) and messages with empty text.
    * Resolves reply_to_message_id to the replied-to text via an id->text map,
      so RagIngestor can inline reply context exactly like the live path.
    * Timestamps come from `date` (ISO local time in the export).

    Returns messages in file order (Telegram exports are already chronological).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(
            f"{path}: no top-level 'messages' array — is this a Telegram JSON export?"
        )

    # First pass: id -> flattened text, to resolve replies.
    text_by_id: Dict[int, str] = {}
    for m in raw_messages:
        mid = m.get("id")
        if isinstance(mid, int):
            text_by_id[mid] = _flatten_text(m.get("text", ""))

    messages: List[ChatMessage] = []
    skipped_service = 0
    skipped_empty = 0

    for m in raw_messages:
        if m.get("type") != "message":
            skipped_service += 1
            continue

        text = _flatten_text(m.get("text", "")).strip()
        if not text:
            skipped_empty += 1
            continue

        date_str = m.get("date")
        try:
            ts = datetime.fromisoformat(date_str)
        except (TypeError, ValueError):
            skipped_empty += 1
            continue

        reply_id = m.get("reply_to_message_id")
        reply_text = text_by_id.get(reply_id) if isinstance(reply_id, int) else None
        if reply_text:
            reply_text = reply_text.strip() or None

        messages.append(
            ChatMessage(
                user_id=_parse_from_id(m.get("from_id")),
                username=str(m.get("from") or "Unknown"),
                text=text,
                message_id=m.get("id", 0) if isinstance(m.get("id"), int) else 0,
                timestamp=ts,
                reply_to_text=reply_text,
            )
        )

    logger.info(
        "Parsed %d messages (skipped %d service, %d empty/undated)",
        len(messages), skipped_service, skipped_empty,
    )
    return messages


# ---------------------------------------------------------------------------
# Grouping by calendar day (reusing the nightly ingest grouping per day)
# ---------------------------------------------------------------------------
def group_by_day(
    messages: List[ChatMessage], tz: str
) -> Dict[date, List[ChatMessage]]:
    """
    Bucket messages by their local calendar day (in the configured timezone).
    Returned dict is ordered oldest-day-first.
    """
    buckets: Dict[date, List[ChatMessage]] = {}
    for msg in messages:
        day = to_aware(msg.timestamp, tz).date()
        buckets.setdefault(day, []).append(msg)
    return dict(sorted(buckets.items()))


# ---------------------------------------------------------------------------
# Progress file (resumability)
# ---------------------------------------------------------------------------
def load_progress(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_days", []))
    except Exception as e:
        logger.warning("Could not read progress file %s: %s", path, e)
        return set()


def save_progress(path: str, completed: set) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"completed_days": sorted(completed)}, f, indent=2)
    except Exception as e:
        logger.warning("Could not write progress file %s: %s", path, e)


# ---------------------------------------------------------------------------
# Insert with retry
# ---------------------------------------------------------------------------
async def _insert_with_retry(
    rag_client: RagClient, text: str, file_source: str
) -> bool:
    for attempt in range(1, _INSERT_MAX_ATTEMPTS + 1):
        try:
            await rag_client.insert(text, file_source=file_source)
            return True
        except RagClientError as e:
            if attempt == _INSERT_MAX_ATTEMPTS:
                logger.error("  insert failed after %d attempts: %s", attempt, e)
                return False
            delay = _INSERT_RETRY_BASE_DELAY * attempt
            logger.warning(
                "  insert attempt %d/%d failed: %s — retrying in %.0fs",
                attempt, _INSERT_MAX_ATTEMPTS, e, delay,
            )
            await asyncio.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Main import routine
# ---------------------------------------------------------------------------
async def run_import(args, config: BotConfig) -> int:
    messages = parse_export(args.file)
    if not messages:
        logger.error("No importable messages found — nothing to do.")
        return 1

    tz = config.timezone
    by_day = group_by_day(messages, tz)

    # Optional month/date-range window (inclusive) for batching.
    since_d = _parse_date_arg(args.since)
    until_d = _parse_date_arg(args.until)
    days = [
        d for d in by_day
        if (since_d is None or d >= since_d) and (until_d is None or d <= until_d)
    ]
    if not days:
        logger.error("No days in the selected range.")
        return 1

    progress_path = args.progress
    completed = set() if args.force else load_progress(progress_path)

    # A RagIngestor purely to reuse group_into_blocks/_render_block. It never
    # touches memory/firebase here, so those are None.
    grouper = RagIngestor(rag_client=None, memory=None, firebase_db=None, config=config)

    logger.info(
        "Importing %d day(s) from %s to %s (%d already done, will skip)",
        len(days), days[0], days[-1], len(completed & set(map(str, days))),
    )

    rag_client: Optional[RagClient] = None
    if not args.dry_run:
        rag_client = _build_rag_client(config)
        if rag_client is None:
            logger.error("LightRAG not configured (.env) — cannot import. Use --dry-run to preview.")
            return 1
        if not await rag_client.health():
            logger.error("LightRAG health check failed — check URL/credentials.")
            return 1

    total_days = 0
    total_blocks = 0
    total_msgs = 0
    failed_days: List[str] = []

    for day in days:
        day_key = str(day)
        if day_key in completed:
            continue

        day_messages = by_day[day]
        blocks = grouper.group_into_blocks(day_messages)
        if not blocks:
            continue

        combined_text = "\n\n".join(blocks)
        file_source = f"telegram_import_{day_key}"

        if args.dry_run:
            logger.info(
                "[DRY] %s: %d msgs -> %d blocks, %d chars (source=%s)",
                day_key, len(day_messages), len(blocks), len(combined_text), file_source,
            )
            if args.show:
                print("-" * 60)
                print(combined_text[:2000])
                print("-" * 60)
        else:
            logger.info(
                "%s: %d msgs -> %d blocks -> inserting...",
                day_key, len(day_messages), len(blocks),
            )
            ok = await _insert_with_retry(rag_client, combined_text, file_source)
            if not ok:
                failed_days.append(day_key)
                continue
            completed.add(day_key)
            save_progress(progress_path, completed)
            # Be gentle: LightRAG extracts+embeds each doc with its own LLM.
            await asyncio.sleep(args.delay)

        total_days += 1
        total_blocks += len(blocks)
        total_msgs += len(day_messages)

    verb = "would import" if args.dry_run else "imported"
    logger.info(
        "Done: %s %d day(s), %d block(s), %d message(s).",
        verb, total_days, total_blocks, total_msgs,
    )
    if failed_days:
        logger.warning(
            "%d day(s) FAILED (not marked done, safe to re-run): %s",
            len(failed_days), ", ".join(failed_days),
        )
        return 2
    return 0


def _parse_date_arg(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        logger.error("Bad date %r — expected YYYY-MM-DD", s)
        sys.exit(2)


def _build_rag_client(config: BotConfig) -> Optional[RagClient]:
    """Same construction as main.build_rag_client, kept local to avoid importing
    the whole bot (and its Telegram/Firebase setup) for a CLI script."""
    if not config.lightrag_enabled or not config.lightrag_api_url:
        return None
    return RagClient(
        base_url=config.lightrag_api_url,
        username=config.lightrag_api_user,
        password=config.lightrag_api_password,
        query_mode=config.lightrag_query_mode,
        query_top_k=config.lightrag_query_top_k,
        query_timeout=config.lightrag_query_timeout,
        insert_timeout=config.lightrag_insert_timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a Telegram JSON export into LightRAG (one doc per day)."
    )
    parser.add_argument("--file", required=True, help="Path to Telegram result.json export")
    parser.add_argument("--since", help="Only import days >= this date (YYYY-MM-DD)")
    parser.add_argument("--until", help="Only import days <= this date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Parse+group only, no inserts")
    parser.add_argument("--show", action="store_true", help="With --dry-run, print block text")
    parser.add_argument("--force", action="store_true", help="Ignore progress file, re-import all days")
    parser.add_argument("--progress", default=_PROGRESS_FILE_DEFAULT, help="Progress file path")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to wait between day inserts")
    args = parser.parse_args()

    config = load_config()
    exit_code = asyncio.run(run_import(args, config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
