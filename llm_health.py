# llm_health.py
"""
Process-wide circuit breaker for "DeepSeek balance exhausted" (HTTP 402).

Why this exists
---------------
On 2026-10-14 the DeepSeek account ran out of balance. Every LLM call then
failed instantly with HTTP 402, but the bot kept going message by message:
the classifier fell back to heuristics, the pipeline still queried LightRAG
for each one, and the generator hit the same 402 at the end (that day's
logs: 181 classifier failures + 13 generation errors, chat got error
replies). Retrieval work was spent on answers that could never be generated.

Behaviour
---------
* Any 402 ("Insufficient Balance") from the classifier or the generator
  arms a cooldown (`_COOLDOWN_S`).
* While armed, analyze_and_respond() short-circuits BEFORE classification
  and BEFORE the LightRAG query — no LLM calls, no retrieval, the bot just
  stays quiet on incoming messages.
* After the cooldown the next message probes the API normally; a working
  balance resumes instantly (the deadline simply expires), a still-empty
  balance re-arms the cooldown. No reset call needed.

Process-local by design (same tradeoff as GrudgeTracker): a restart is
fine because the first message after it re-detects the 402 and re-arms.
"""

import time
from typing import Optional

# How long to stay quiet before probing the API again while the balance is
# empty. Long enough not to hammer a broken account, short enough that a
# top-up is noticed within minutes (on the next incoming message).
_COOLDOWN_S = 300.0

# time.monotonic() deadline until which the pipeline stays paused; 0 = healthy.
_exhausted_until = 0.0


def is_insufficient_balance(exc: Exception) -> bool:
    """
    True for DeepSeek HTTP 402 / "Insufficient Balance" errors.

    Covers both call sites: the classifier raises DeepSeekHTTPError
    (retry.py) carrying `.status`, the generator uses the OpenAI SDK whose
    errors carry `.status_code` and bake the JSON body into str(exc).
    """
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 402:
        return True
    return "Insufficient Balance" in str(exc)


def mark_balance_exhausted(now: Optional[float] = None) -> None:
    """Arm the cooldown after a 402 was seen (now defaults to monotonic)."""
    global _exhausted_until
    _exhausted_until = (now if now is not None else time.monotonic()) + _COOLDOWN_S


def balance_exhausted(now: Optional[float] = None) -> bool:
    """True while the pause is active (i.e. the last 402 is within cooldown)."""
    return (now if now is not None else time.monotonic()) < _exhausted_until


def seconds_left(now: Optional[float] = None) -> float:
    """Seconds until the next probe is allowed (0 when healthy)."""
    return max(0.0, _exhausted_until - (now if now is not None else time.monotonic()))


def cooldown_seconds() -> float:
    """The configured cooldown, exposed for log messages and tests."""
    return _COOLDOWN_S
