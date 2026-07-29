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
* Long pasted blobs are truncated (--max-msg-chars) so copied prompts/articles
  don't end up retrieved as "facts about people".
* Media messages with no caption become a bare "[фото]" / "[стикер]" line, so
  the conversation reacting to them still makes sense. Days made up of nothing
  but media are skipped — there is no fact to extract from them.
* Sender identity is keyed on Telegram's from_id via a persisted name map
  (--name-map), so a friend renaming themselves between exports does not turn
  into a second person in the graph.
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
from collections import Counter, defaultdict
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
_NAME_MAP_FILE_DEFAULT = "import_names.json"
_INSERT_MAX_ATTEMPTS = 3
_INSERT_RETRY_BASE_DELAY = 2.0  # seconds; backfill is not latency-sensitive

# Long pasted blobs (jailbreak prompts, copied articles, code dumps) are a
# problem for a *chat* knowledge base: LightRAG keeps raw chunks alongside the
# extracted graph, so a 2686-char pasted prompt gets retrieved later and served
# to the bot as a "fact about a person". In the real export these blobs were
# 52% of all text while the median message was 16 chars. Truncating keeps the
# useful signal ("Кирилл pasted a jailbreak prompt") and drops the noise.
_MAX_MSG_CHARS_DEFAULT = 500
_TRUNCATION_MARKER = " […обрезано]"

# Telegram records the bot's messages under its @username display name
# ("DeepSeek"), but the live bot introduces itself in prompts/ingest as
# BOT_NAME ("Дип Сик"). Left as-is, LightRAG would build two separate entities
# for the same actor and split every fact about the bot between them. Map the
# export spelling(s) onto the configured name at parse time.
_SENDER_ALIASES = {
    "DeepSeek": None,      # None = replace with config.bot_name
    "Deep Seek": None,
    "deepseek": None,
}

# A message with no text is still an event: someone sent a photo, and the next
# three messages discuss it. Dropped entirely (the old behaviour), LightRAG saw
# the discussion without its subject — and four days that held nothing but
# media never got a document at all. A bare word is enough to restore the link;
# we have no description of the content and don't invent one.
_MEDIA_LABELS = {
    "sticker": "стикер",
    "animation": "гифка",
    "video_message": "кружок",
    "video_file": "видео",
    "voice_message": "голосовое",
    "audio_file": "аудио",
}
_MEDIA_LABEL_PHOTO = "фото"
_MEDIA_LABEL_FALLBACK = "файл"


def _media_placeholder(m: dict) -> Optional[str]:
    """Return "[фото]"-style stand-in text for a media message, or None."""
    media_type = m.get("media_type")
    if media_type:
        label = _MEDIA_LABELS.get(media_type, _MEDIA_LABEL_FALLBACK)
    elif m.get("photo"):
        label = _MEDIA_LABEL_PHOTO
    elif m.get("file"):
        label = _MEDIA_LABEL_FALLBACK
    else:
        return None
    return f"[{label}]"


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


def _resolve_sender(raw_name: str, bot_name: Optional[str]) -> str:
    """
    Normalise a sender name from the export.

    The only rewrite today is the bot's own display name -> BOT_NAME, so the
    knowledge graph has one bot entity instead of two (see _SENDER_ALIASES).
    """
    if raw_name in _SENDER_ALIASES and bot_name:
        return _SENDER_ALIASES[raw_name] or bot_name
    return raw_name


# ---------------------------------------------------------------------------
# Canonical sender names (stable across exports)
# ---------------------------------------------------------------------------
# Telegram exports carry whatever display name a person had at export time. A
# friend who renames themselves between two exports would land in the graph as
# a second person, and every fact about them would be split — the same mess we
# had to merge by hand for Тима/Tima. from_id never changes, so it is the real
# identity. The map is persisted: the FIRST name we ever saw for an id wins,
# including in imports run months later.
def build_name_map(
    messages_raw: List[dict], known: Dict[str, str], bot_name: Optional[str]
) -> Dict[str, str]:
    """
    Extend `known` (from_id -> canonical name) with ids seen in this export.

    Ids already in `known` keep their name. New ids take the name they use most
    often in the export.
    """
    seen: Dict[str, Counter] = defaultdict(Counter)
    for m in messages_raw:
        if m.get("type") != "message":
            continue
        from_id = str(m.get("from_id") or "")
        if not from_id:
            continue
        seen[from_id][_resolve_sender(str(m.get("from") or "Unknown"), bot_name)] += 1

    name_map = dict(known)
    for from_id, names in seen.items():
        if from_id not in name_map:
            name_map[from_id] = names.most_common(1)[0][0]
    return name_map


def load_name_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f).items()}
    except Exception as e:
        logger.warning("Could not read name map %s: %s", path, e)
        return {}


def save_name_map(path: str, name_map: Dict[str, str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(name_map, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Could not write name map %s: %s", path, e)


def parse_export(
    path: str,
    max_msg_chars: int = _MAX_MSG_CHARS_DEFAULT,
    bot_name: Optional[str] = None,
    name_map: Optional[Dict[str, str]] = None,
) -> List[ChatMessage]:
    """
    Parse a Telegram Desktop JSON export into ChatMessage objects.

    * Skips service messages (joins/pins/calls).
    * Media without a caption becomes "[фото]" / "[стикер]" etc. so the
      conversation around it stays intact (see _MEDIA_LABELS).
    * Resolves reply_to_message_id to the replied-to text via an id->text map,
      so RagIngestor can inline reply context exactly like the live path.
    * Timestamps come from `date` (ISO local time in the export).
    * Truncates messages longer than `max_msg_chars` (0 = keep everything) —
      see _MAX_MSG_CHARS_DEFAULT for why.
    * Names come from `name_map` (from_id -> canonical name) when given, so a
      renamed friend stays one person across exports.

    Returns messages in file order (Telegram exports are already chronological).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(
            f"{path}: no top-level 'messages' array — is this a Telegram JSON export?"
        )

    # First pass: id -> flattened text, to resolve replies. Media placeholders
    # are included so a reply to a photo quotes "[фото]" instead of nothing.
    text_by_id: Dict[int, str] = {}
    for m in raw_messages:
        mid = m.get("id")
        if isinstance(mid, int):
            text_by_id[mid] = _flatten_text(m.get("text", "")).strip() or (
                _media_placeholder(m) or ""
            )

    name_map = name_map or {}
    messages: List[ChatMessage] = []
    skipped_service = 0
    skipped_empty = 0
    truncated = 0
    chars_dropped = 0
    renamed = 0
    media_kept = 0

    for m in raw_messages:
        if m.get("type") != "message":
            skipped_service += 1
            continue

        text = _flatten_text(m.get("text", "")).strip()
        if not text:
            placeholder = _media_placeholder(m)
            if placeholder is None:
                skipped_empty += 1
                continue
            text = placeholder
            media_kept += 1

        if max_msg_chars and len(text) > max_msg_chars:
            chars_dropped += len(text) - max_msg_chars
            truncated += 1
            text = text[:max_msg_chars].rstrip() + _TRUNCATION_MARKER

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

        raw_sender = str(m.get("from") or "Unknown")
        from_id = str(m.get("from_id") or "")
        sender = name_map.get(from_id) or _resolve_sender(raw_sender, bot_name)
        if sender != raw_sender:
            renamed += 1

        messages.append(
            ChatMessage(
                user_id=_parse_from_id(m.get("from_id")),
                username=sender,
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
    if media_kept:
        logger.info("Kept %d media message(s) as placeholders ([фото], [стикер], ...)", media_kept)
    if truncated:
        logger.info(
            "Truncated %d long message(s) at %d chars — %d chars of noise dropped",
            truncated, max_msg_chars, chars_dropped,
        )
    if renamed:
        logger.info("Normalised %d sender name(s)", renamed)
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


def _is_media_only(day_messages: List[ChatMessage]) -> bool:
    """
    True when a day holds nothing but media placeholders.

    "Максим: [фото] / Кирилл: [фото]" gives the extractor nothing to learn, so
    the day is skipped rather than inserted as an empty document. Placeholders
    mixed WITH real text are kept — there they anchor the conversation.
    """
    return all(
        m.text.startswith("[") and m.text.endswith("]") and len(m.text) < 15
        for m in day_messages
    )


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
    with open(args.file, "r", encoding="utf-8") as f:
        raw = json.load(f).get("messages") or []

    name_map = build_name_map(raw, load_name_map(args.name_map), config.bot_name)
    if not args.dry_run:
        save_name_map(args.name_map, name_map)
    logger.info("Sender identities: %d known (%s)", len(name_map), args.name_map)

    messages = parse_export(
        args.file,
        max_msg_chars=args.max_msg_chars,
        bot_name=config.bot_name,
        name_map=name_map,
    )
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
    skipped_media_days = 0
    failed_days: List[str] = []

    for day in days:
        day_key = str(day)
        if day_key in completed:
            continue

        day_messages = by_day[day]
        if _is_media_only(day_messages):
            logger.info("%s: %d media message(s), nothing to extract — skipped", day_key, len(day_messages))
            skipped_media_days += 1
            continue

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
    if skipped_media_days:
        logger.info("Skipped %d day(s) containing only media.", skipped_media_days)
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
    parser.add_argument(
        "--name-map", default=_NAME_MAP_FILE_DEFAULT, dest="name_map",
        help="from_id -> canonical name map, so renamed people stay one entity",
    )
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to wait between day inserts")
    parser.add_argument(
        "--max-msg-chars", type=int, default=_MAX_MSG_CHARS_DEFAULT, dest="max_msg_chars",
        help=f"Truncate messages longer than this (default {_MAX_MSG_CHARS_DEFAULT}, 0 = keep full text)",
    )
    args = parser.parse_args()

    config = load_config()
    exit_code = asyncio.run(run_import(args, config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
