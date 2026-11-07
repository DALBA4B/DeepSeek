# rag_cache.py
"""
Short-lived in-RAM cache for LightRAG context.

Why
---
Follow-up questions in the same thread often trigger another /query within
seconds, and every query adds a large retrieved context to the next prompt.
Reusing the previous context for a genuinely related follow-up removes that
duplicate spend without changing how the bot answers.

Safety
------
Deliberately conservative: one entry per chat, an absolute TTL (the entry is
never extended by a hit), and reuse only when BOTH are true:

* the same person is asking again, or the message is in the same reply thread;
* the new query overlaps the cached one (or is exactly the same query).

A stale or unrelated context is worse than a fresh query, so anything
ambiguous is a cache miss. Process-local only: this resets on restart, the
same tradeoff as the other RAM trackers (grudge, recent responses).
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

from rerank import content_words, norm

# Reuse requires at least this many shared content words...
DEFAULT_MIN_OVERLAP = 2
# ...and this share of the shorter query's words.
DEFAULT_MIN_CONTAINMENT = 0.6
# Two minutes: long enough to catch a burst of follow-ups, short enough that
# a topic change cannot keep an old context alive for long.
DEFAULT_TTL_SECONDS = 120.0


@dataclass
class _Entry:
    created_at: float
    user_id: Optional[int]
    author_norm: str
    message_norm: str
    reply_norm: str
    query_norm: str
    query_words: frozenset
    facts: str


class RagCache:
    """One recent LightRAG result per chat, reused only for close follow-ups."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        min_overlap: int = DEFAULT_MIN_OVERLAP,
        min_containment: float = DEFAULT_MIN_CONTAINMENT,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.min_overlap = min_overlap
        self.min_containment = min_containment
        self._entries: Dict[int, _Entry] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        chat_id: int,
        author: str,
        user_id: Optional[int],
        message_text: str,
        reply_text: Optional[str],
        query: str,
        now: Optional[float] = None,
    ) -> Optional[str]:
        """Return cached facts for this follow-up, or None on any doubt."""
        entry = self._entries.get(chat_id)
        if entry is None:
            self.misses += 1
            return None

        now = time.monotonic() if now is None else now
        age = now - entry.created_at
        if age < 0 or age > self.ttl_seconds:
            # Absolute TTL: a hit never extends the entry's life.
            self._entries.pop(chat_id, None)
            self.misses += 1
            return None

        same_author = self._same_author(entry, author, user_id)
        same_thread = self._same_thread(entry, message_text, reply_text)
        if not (same_author or same_thread):
            self.misses += 1
            return None

        if not self._queries_match(entry, query):
            self.misses += 1
            return None

        self.hits += 1
        return entry.facts

    def put(
        self,
        *,
        chat_id: int,
        author: str,
        user_id: Optional[int],
        message_text: str,
        reply_text: Optional[str],
        query: str,
        facts: str,
        now: Optional[float] = None,
    ) -> None:
        """Store one successful retrieval, replacing the chat's previous entry."""
        if not facts:
            return
        now = time.monotonic() if now is None else now
        self._entries[chat_id] = _Entry(
            created_at=now,
            user_id=user_id,
            author_norm=norm(author or ""),
            message_norm=norm(message_text or ""),
            reply_norm=norm(reply_text or ""),
            query_norm=norm(query.strip()),
            query_words=frozenset(content_words(query)),
            facts=facts,
        )

    def invalidate(self, chat_id: int) -> None:
        """Drop a chat's entry (e.g. on /off or a manual memory refresh)."""
        self._entries.pop(chat_id, None)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _same_author(entry: _Entry, author: str, user_id: Optional[int]) -> bool:
        if user_id is not None and entry.user_id is not None:
            return user_id == entry.user_id
        author_norm = norm(author or "")
        return bool(author_norm) and author_norm == entry.author_norm

    @staticmethod
    def _same_thread(
        entry: _Entry, message_text: str, reply_text: Optional[str]
    ) -> bool:
        current_reply = norm(reply_text or "")
        if not current_reply:
            return False
        # The new message replies directly to the message that triggered the
        # cached query...
        if current_reply == entry.message_norm:
            return True
        # ...or both messages are replies to the same parent (same thread).
        return bool(entry.reply_norm) and current_reply == entry.reply_norm

    def _queries_match(self, entry: _Entry, query: str) -> bool:
        query_norm = norm(query.strip())
        if query_norm == entry.query_norm:
            return True

        new_words = content_words(query)
        old_words = entry.query_words
        if not new_words or not old_words:
            return False

        shared = new_words & old_words
        if len(shared) < self.min_overlap:
            return False
        shorter = min(len(new_words), len(old_words))
        return len(shared) / shorter >= self.min_containment
