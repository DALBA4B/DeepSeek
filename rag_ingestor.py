# rag_ingestor.py
"""
Nightly pipeline that feeds the day's chat into LightRAG.

Why this exists
---------------
LightRAG only "knows" what we send it. Real-time messages are NOT inserted on
arrival (that would be slow and noisy). Instead, once a night we gather the
chat, group it into coherent conversation blocks, and hand those blocks to
LightRAG, which extracts entities/relations (via its own DeepSeek) and embeds
them (via its own OpenAI). The bot later queries those facts at answer time.

The grouping is the important part. A flat dump of messages loses the link
between "я люблю киев" and the "хахаха +1" that agrees with it — LightRAG's
extractor would index them as unrelated facts. By grouping messages that are
close in time (and inlining the replied-to text), the extractor sees the
conversation as it actually happened.

Idempotency
-----------
Restarts/redeploys are common on Railway/Render. To avoid re-inserting the
same day twice (and to avoid skipping a day if the bot was down at ingest
time), the last successful ingest timestamp is persisted in Firebase under
collection `rag_meta` / doc `ingest_state`. Each run resumes from there.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from models import BotConfig, ChatMessage
from rag_client import RagClient
from utils import get_now, to_aware

logger = logging.getLogger(__name__)

# Firebase document that stores where the last ingest got to.
_RAG_META_COLLECTION = "rag_meta"
_RAG_META_DOC = "ingest_state"
_LAST_INGEST_FIELD = "last_ingest_timestamp"
_LAST_STATS_FIELD = "last_ingest_stats"


class RagIngestor:
    """
    Collect → group → insert chat history into LightRAG.

    Attributes:
        rag_client:     Live RagClient (insert() is called per block).
        memory:         Memory instance (used for the RAM fallback source).
        firebase_db:    Optional Firestore client for the durable source and
                        for persisting the ingest cursor.
        config:         BotConfig (grouping params + timezone).
    """

    def __init__(
        self,
        rag_client: RagClient,
        memory,
        firebase_db=None,
        config: Optional[BotConfig] = None,
    ) -> None:
        self.rag_client = rag_client
        self.memory = memory
        self.firebase_db = firebase_db
        self.config = config
        self._tz = config.timezone if config is not None else "UTC"

    # ------------------------------------------------------------------ #
    # Cursor (idempotency): where did the last ingest stop?
    # ------------------------------------------------------------------ #
    def get_last_ingest_timestamp(self) -> Optional[datetime]:
        """
        Read the persisted "last successful ingest" timestamp from Firebase.

        Returns:
            Timezone-aware datetime of the last ingest end, or None if no
            previous ingest is recorded (first run / Firebase unavailable).
        """
        if self.firebase_db is None:
            return None
        try:
            doc = (
                self.firebase_db.collection(_RAG_META_COLLECTION)
                .document(_RAG_META_DOC)
                .get()
            )
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            ts = data.get(_LAST_INGEST_FIELD)
            if isinstance(ts, str):
                try:
                    return to_aware(datetime.fromisoformat(ts), self._tz)
                except ValueError:
                    return None
            if isinstance(ts, datetime):
                return to_aware(ts, self._tz)
            return None
        except Exception as e:
            logger.warning(f"Could not read last ingest timestamp: {e}")
            return None

    def _save_last_ingest_timestamp(self, ts: datetime, stats: dict) -> None:
        """Persist the ingest cursor + a small stats snapshot for /ragstats."""
        if self.firebase_db is None:
            return
        try:
            self.firebase_db.collection(_RAG_META_COLLECTION).document(
                _RAG_META_DOC
            ).set(
                {
                    _LAST_INGEST_FIELD: ts.isoformat(),
                    _LAST_STATS_FIELD: stats,
                },
                merge=True,
            )
        except Exception as e:
            logger.warning(f"Could not save last ingest timestamp: {e}")

    def get_last_stats(self) -> Optional[dict]:
        """Return the stats dict from the most recent successful ingest."""
        if self.firebase_db is None:
            return None
        try:
            doc = (
                self.firebase_db.collection(_RAG_META_COLLECTION)
                .document(_RAG_META_DOC)
                .get()
            )
            if not doc.exists:
                return None
            return (doc.to_dict() or {}).get(_LAST_STATS_FIELD)
        except Exception as e:
            logger.warning(f"Could not read last ingest stats: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Collect
    # ------------------------------------------------------------------ #
    def collect_messages(self, since: Optional[datetime] = None) -> List[ChatMessage]:
        """
        Gather all user messages since `since` (or for the last 24h if None).

        Delegates to Memory.get_messages_for_period, which prefers Firebase
        and falls back to the in-RAM daily_log.
        """
        if since is None:
            since = get_now(self._tz) - timedelta(hours=24)
        return self.memory.get_messages_for_period(since=since)

    # ------------------------------------------------------------------ #
    # Group into time-coherent blocks (THE key step)
    # ------------------------------------------------------------------ #
    def group_into_blocks(
        self,
        messages: List[ChatMessage],
        block_gap_minutes: Optional[int] = None,
        max_per_block: Optional[int] = None,
    ) -> List[str]:
        """
        Split sorted messages into text blocks that read like a conversation.

        A new block starts when:
          * the gap to the previous message exceeds `block_gap_minutes`, OR
          * the current block already has `max_per_block` messages.

        Each block is rendered as:
            [14:30-14:42]
            Вася: я люблю киев
            Петя (reply на «я люблю киев»): согласен
            Максим: +

        Args:
            messages:       Messages to group (will be sorted by timestamp).
            block_gap_minutes: Override for config.rag_ingest_timeblock_minutes.
            max_per_block:  Override for config.rag_ingest_max_per_block.

        Returns:
            List of block strings ready for RagClient.insert().
        """
        if not messages:
            return []

        if block_gap_minutes is None:
            block_gap_minutes = (
                self.config.rag_ingest_timeblock_minutes if self.config else 15
            )
        if max_per_block is None:
            max_per_block = self.config.rag_ingest_max_per_block if self.config else 30

        ordered = sorted(messages, key=lambda m: m.timestamp)
        gap = timedelta(minutes=block_gap_minutes)

        blocks: List[List[ChatMessage]] = []
        current: List[ChatMessage] = []
        prev_ts: Optional[datetime] = None

        for msg in ordered:
            ts = to_aware(msg.timestamp, self._tz)
            # Start a new block on a time gap OR a size cap
            if current and (
                (prev_ts is not None and (ts - prev_ts) > gap)
                or len(current) >= max_per_block
            ):
                blocks.append(current)
                current = []
            current.append(msg)
            prev_ts = ts
        if current:
            blocks.append(current)

        return [self._render_block(b) for b in blocks]

    @staticmethod
    def _render_block(block: List[ChatMessage]) -> str:
        """
        Render a message group as one readable text block for LightRAG.
        Reply context is inlined so the extractor sees the link between a
        message and what it reacts to.
        """
        start = block[0].timestamp
        end = block[-1].timestamp
        header = f"[{start.strftime('%H:%M')}-{end.strftime('%H:%M')}]"
        lines = [header]
        for msg in block:
            quote = (msg.reply_to_text or "").strip().replace("\n", " ")
            if len(quote) > 80:
                quote = quote[:77] + "..."
            if quote:
                lines.append(f"{msg.username} (reply на «{quote}»): {msg.text}")
            else:
                lines.append(f"{msg.username}: {msg.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # The full pipeline
    # ------------------------------------------------------------------ #
    async def ingest(self, since: Optional[datetime] = None) -> dict:
        """
        Run the full nightly ingest: collect → group → insert block by block.

        On success the ingest cursor is advanced so the next run resumes from
        "now". LightRAG failures on individual blocks are logged but do not
        abort the whole run — partial progress is still persisted.

        Args:
            since: Start of the window to ingest. Defaults to the last cursor
                (or 24h ago on the very first run).

        Returns:
            Stats dict: {blocks, messages, inserted, failed, started_at, since}
        """
        started_at = get_now(self._tz)
        # Resolve the window start from the persisted cursor when not given.
        if since is None:
            since = self.get_last_ingest_timestamp()
            if since is None:
                since = started_at - timedelta(hours=24)

        messages = self.collect_messages(since=since)
        if not messages:
            logger.info("RAG ingest: no new messages since %s", since)
            return {
                "blocks": 0,
                "messages": 0,
                "inserted": 0,
                "failed": 0,
                "since": since.isoformat(),
                "started_at": started_at.isoformat(),
                "skipped": "no_messages",
            }

        blocks = self.group_into_blocks(messages)
        logger.info(
            "RAG ingest: %d messages → %d blocks (since=%s)",
            len(messages),
            len(blocks),
            since,
        )

        inserted = 0
        failed = 0
        ids: List[Optional[str]] = []
        for i, block in enumerate(blocks, 1):
            try:
                doc_id = await self.rag_client.insert(block)
                if doc_id is not None:
                    ids.append(doc_id)
                inserted += 1
                logger.debug("RAG ingest: block %d/%d inserted", i, len(blocks))
            except Exception as e:
                # A single block failing must not kill the whole nightly run.
                failed += 1
                logger.warning("RAG ingest: block %d failed: %s", i, e)

        stats = {
            "blocks": len(blocks),
            "messages": len(messages),
            "inserted": inserted,
            "failed": failed,
            "since": since.isoformat(),
            "started_at": started_at.isoformat(),
            "finished_at": get_now(self._tz).isoformat(),
        }

        # Only advance the cursor if we actually made progress, so a total
        # outage doesn't silently skip a day's worth of messages.
        if inserted > 0:
            self._save_last_ingest_timestamp(started_at, stats)

        logger.info("RAG ingest done: %s", stats)
        return stats
