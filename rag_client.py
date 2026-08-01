# rag_client.py
"""
LightRAG REST API client for the bot.

LightRAG acts as the bot's long-term knowledge base about people and
facts discussed in the chat. This client is a thin async wrapper over
the LightRAG HTTP API:

    - retrieve(): query facts (used by brain.py before answering)
    - insert():   nightly ingest of chat blocks (used by rag_ingestor.py)
    - clear():    wipe storage (debug / reset)
    - health():   connectivity check (used by /ragstats)

Authentication
--------------
LightRAG does NOT accept HTTP Basic auth. It uses JWT tokens:
    1. POST /login with form-encoded username+password
    2. Receive an {access_token, ...} JSON
    3. Send it as `Authorization: Bearer <token>` on every call
This client caches the token, refreshes it before expiry, and retries
once automatically on a 401 (token may have been revoked server-side).

Design notes
------------
* All network calls go through aiohttp (the bot is fully async).
* Every call is wrapped with a timeout + try/except: a LightRAG outage
  must NEVER crash the bot. On failure we degrade gracefully (no facts).
* LightRAG is configured separately on Railway with its own models
  (LLM = DeepSeek for extraction, Embeddings = OpenAI). The bot only
  talks to it over HTTP, so no API keys for DeepSeek/OpenAI live here.
"""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)
_rag_debug = logging.getLogger("ragdebug")

# Refresh the token a bit before it actually expires, to avoid edge-case 401s
_TOKEN_REFRESH_MARGIN_SECONDS = 60


class RagClientError(Exception):
    """Non-fatal error from LightRAG (logged, but does not crash the bot)."""


class RagClient:
    """
    Async client for a remote LightRAG server.

    Attributes:
        base_url:       LightRAG service URL (e.g. https://...up.railway.app)
        username:       Auth user (must match AUTH_ACCOUNTS on LightRAG)
        password:       Auth password
        query_mode:     Retrieval mode (mix / hybrid / local / global / naive)
        query_top_k:    How many chunks/entities to retrieve per query
        query_max_tokens: Cap on the size of the returned context (0 = no cap).
            LightRAG otherwise returns everything it finds — tens of thousands
            of characters that all end up in the prompt.
        query_timeout:  Seconds to wait for a query answer
        insert_timeout: Seconds to wait for an insert acknowledgement
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        query_mode: str = "mix",
        query_top_k: int = 5,
        query_max_tokens: int = 14000,
        query_timeout: float = 35.0,
        insert_timeout: float = 15.0,
    ) -> None:
        # Normalize: strip trailing slash, drop accidental /webui suffix
        url = base_url.strip().rstrip("/")
        if url.endswith("/webui"):
            url = url[: -len("/webui")]
        self.base_url = url
        self.username = username
        self.password = password
        self.query_mode = query_mode
        self.query_top_k = query_top_k
        self.query_max_tokens = query_max_tokens
        self.query_timeout = query_timeout
        self.insert_timeout = insert_timeout

        # Reason the last retrieve() failed (timeout / network / HTTP error),
        # or None if it succeeded or simply found nothing. brain.py reads this
        # to tell "LightRAG is down" apart from "nothing relevant in the graph".
        self.last_error: Optional[str] = None

        # Cached JWT state
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._login_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _login(self) -> None:
        """
        Authenticate against LightRAG and cache the JWT token.
        Raises RagClientError on failure.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # LightRAG uses OAuth2PasswordRequestForm: form-encoded body
                async with session.post(
                    self._url("/login"),
                    data={"username": self.username, "password": self.password},
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RagClientError(
                            f"LightRAG login -> {resp.status}: {text[:300]}"
                        )
                    data = await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            raise RagClientError(f"Network error during LightRAG login: {e}") from e
        except RagClientError:
            raise
        except Exception as e:
            raise RagClientError(f"Unexpected error during LightRAG login: {e}") from e

        token = data.get("access_token")
        if not token:
            raise RagClientError(f"LightRAG login: no access_token in response: {data}")

        # Token TTL is reported in seconds by LightRAG; fall back to 1h.
        ttl = data.get("expires_in")
        try:
            ttl = float(ttl) if ttl is not None else 3600.0
        except (TypeError, ValueError):
            ttl = 3600.0

        self._token = token
        self._token_expires_at = time.time() + ttl - _TOKEN_REFRESH_MARGIN_SECONDS
        logger.debug("LightRAG login OK, token TTL=%.0fs", ttl)

    async def _ensure_token(self) -> str:
        """
        Return a valid JWT, logging in first if needed.
        Concurrent callers are serialized through a lock to avoid duplicate
        logins when many messages arrive at once.
        """
        if self._token and time.time() < self._token_expires_at:
            return self._token

        async with self._login_lock:
            # Re-check inside the lock: another caller may have just refreshed
            if self._token and time.time() < self._token_expires_at:
                return self._token
            await self._login()
            return self._token  # type: ignore[return-value]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        timeout: float,
        allow_retry: bool = True,
    ) -> dict:
        """
        Perform an authenticated HTTP request to LightRAG.
        Retries once on 401 (token may have expired server-side).
        Raises RagClientError on any failure.
        """
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    self._url(path),
                    json=json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    # Token expired/revoked server-side -> refresh and retry once
                    if resp.status == 401 and allow_retry:
                        logger.debug("LightRAG 401 on %s %s, refreshing token", method, path)
                        self._token = None
                        return await self._request(
                            method, path, json=json, timeout=timeout, allow_retry=False
                        )

                    text = await resp.text()
                    if resp.status >= 400:
                        raise RagClientError(
                            f"LightRAG {method} {path} -> {resp.status}: {text[:300]}"
                        )
                    if not text:
                        return {}
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        return {"raw": text}
        except asyncio.TimeoutError as e:
            raise RagClientError(
                f"LightRAG timed out after {timeout:.1f}s on {method} {path}"
            ) from e
        except aiohttp.ClientError as e:
            raise RagClientError(f"Network error talking to LightRAG: {e}") from e
        except RagClientError:
            raise
        except Exception as e:
            raise RagClientError(f"Unexpected error talking to LightRAG: {e}") from e

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def health(self) -> bool:
        """
        Light connectivity check. Returns True if LightRAG answers at all
        (login succeeds). Used by /ragstats and at startup.
        """
        try:
            await self._ensure_token()
            return True
        except RagClientError as e:
            logger.warning(f"LightRAG health check failed: {e}")
            return False

    async def retrieve(self, query: str) -> Optional[str]:
        """
        Retrieve facts relevant to a natural-language query.

        Uses only_need_context=True so LightRAG returns the raw retrieved
        context (entities + relations + chunks) WITHOUT generating its own
        answer. The bot's DeepSeek then formulates the final chat reply from
        those facts. This keeps LightRAG fast and the "voice" consistent.

        Args:
            query: Natural-language search (e.g. "Максим Dota игры").

        Returns:
            A text block of relevant facts, or None if retrieval failed /
            nothing relevant was found. Callers treat None as "answer without
            extra knowledge" — the bot must never block on LightRAG.
        """
        if not query or not query.strip():
            return None

        self.last_error = None
        payload = {
            "query": query,
            "mode": self.query_mode,
            "only_need_context": True,
            "top_k": self.query_top_k,
        }
        if self.query_max_tokens:
            payload["max_total_tokens"] = self.query_max_tokens
        _rag_debug.info("RAG-DEBUG [lightrag http] POST /query payload=%s", payload)

        t0 = time.monotonic()
        try:
            data = await self._request(
                "POST", "/query", json=payload, timeout=self.query_timeout
            )
        except RagClientError as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.last_error = str(e)
            logger.warning(f"LightRAG retrieve failed (query={query!r}): {e}")
            _rag_debug.info(
                "RAG-DEBUG [lightrag http] /query failed after %.0fms: %s", elapsed_ms, e
            )
            return None
        elapsed_ms = (time.monotonic() - t0) * 1000
        _rag_debug.info(
            "RAG-DEBUG [lightrag http] /query responded in %.0fms, raw keys=%s",
            elapsed_ms, list(data.keys()),
        )

        # LightRAG returns the assembled context in different shapes across
        # versions. Normalize to a single string.
        context = (
            data.get("response")
            or data.get("context")
            or data.get("data")
            or ""
        )
        if isinstance(context, list):
            context = "\n".join(str(x) for x in context)
        context = str(context).strip()

        if not context:
            _rag_debug.info("RAG-DEBUG [lightrag http] normalized context is empty")
            return None
        return context

    async def insert(self, text: str, file_source: str = "telegram_chat") -> Optional[str]:
        """
        Insert a block of text into the knowledge base. LightRAG will chunk
        it, extract entities/relations (via its configured LLM) and embed it
        (via its configured embedding model) asynchronously.

        Args:
            text: A coherent block of chat (see rag_ingestor.py for how blocks
                  are built — grouped by time, with reply context inlined).
            file_source: Label shown as the doc's source in LightRAG's UI.
                This server rejects inserts without a non-empty file_source.

        Returns:
            A track/document id if LightRAG returned one, else None.
        """
        if not text or not text.strip():
            return None

        payload = {"text": text, "file_source": file_source}

        # Let RagClientError propagate — rag_ingestor.ingest() catches it per
        # block to count `failed` correctly (swallowing it here would make
        # every failed insert look like a success).
        data = await self._request(
            "POST", "/documents/text", json=payload, timeout=self.insert_timeout
        )

        # Track id is useful for the ingestor's idempotency log
        return (
            data.get("track_id")
            or data.get("id")
            or data.get("document_id")
            or (data.get("data", {}) or {}).get("id")
        )

