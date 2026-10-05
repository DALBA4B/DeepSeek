# tools/dump_retrieval.py
"""
Record bot-identical LightRAG retrieval outputs for offline rerank experiments.

Read-only against the graph: one /query per question with the exact payload
the bot sends (mode / top_k / max_total_tokens from .env). Writes one JSON
per question into dumps/: the raw server context plus what compact_context()
would forward into the prompt.

Run where .env + deps exist (VPS: /opt/bot):

    python tools/dump_retrieval.py
    python tools/dump_retrieval.py "свой вопрос" --file my_questions.txt

Then copy dumps/ to the notebook for offline experiments (nothing in dumps/
may be committed — chat contents, see .gitignore).

Cost: each NEW question costs one small DeepSeek keyword-extraction call
inside LightRAG (mix mode). Repeats hit the server LLM cache. A dozen
questions is cents.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from rag_client import RagClient, compact_context  # noqa: E402

# Seed set: real past cases (empty retrieve, bloated retrieve, missed answer).
# Extend with your own "бот лажал с памятью" questions via CLI args / --file.
DEFAULT_QUESTIONS = [
    "Рома",
    "Максим Тян чем занимался",
    "Тимур",
    "какая самая лучшая бывшая у максима",
    "Денис",
    "Зазик",
    "что Максим говорил про кальян",
    "кто такой Тимур",
]


def build_client() -> RagClient:
    cfg = load_config()
    return RagClient(
        base_url=cfg.lightrag_api_url,
        username=cfg.lightrag_api_user,
        password=cfg.lightrag_api_password,
        query_mode=cfg.lightrag_query_mode,
        query_top_k=cfg.lightrag_query_top_k,
        query_max_tokens=cfg.lightrag_query_max_tokens,
        query_timeout=cfg.lightrag_query_timeout,
        insert_timeout=cfg.lightrag_insert_timeout,
        context_desc_cap=cfg.lightrag_context_desc_cap,
        context_max_chunks=cfg.lightrag_context_max_chunks,
    )


async def dump_one(rag: RagClient, question: str) -> dict:
    payload = {
        "query": question,
        "mode": rag.query_mode,
        "only_need_context": True,
        "top_k": rag.query_top_k,
    }
    if rag.query_max_tokens:
        payload["max_total_tokens"] = rag.query_max_tokens
    t0 = time.monotonic()
    data = await rag._request("POST", "/query", json=payload, timeout=rag.query_timeout)
    elapsed_ms = (time.monotonic() - t0) * 1000
    context = data.get("response") or data.get("context") or data.get("data") or ""
    if isinstance(context, list):
        context = "\n".join(str(x) for x in context)
    context = str(context).strip()
    compacted, stats = compact_context(context, rag.context_desc_cap, rag.context_max_chunks)
    return {
        "question": question,
        "mode": rag.query_mode,
        "top_k": rag.query_top_k,
        "max_total_tokens": rag.query_max_tokens,
        "desc_cap": rag.context_desc_cap,
        "max_chunks": rag.context_max_chunks,
        "elapsed_ms": round(elapsed_ms, 1),
        "raw_chars": len(context),
        "stats": stats,
        "raw_context": context,
        "compacted": compacted,
    }


async def main_async(questions: list, out_dir: Path) -> int:
    rag = build_client()
    if not await rag.health():
        print("LightRAG unavailable — check .env", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(questions, 1):
        q = q.strip()
        if not q:
            continue
        try:
            rec = await dump_one(rag, q)
        except Exception as e:  # noqa: BLE001 — one bad question must not kill the batch
            print(f"[{i}/{len(questions)}] {q!r} FAILED: {e}")
            continue
        path = out_dir / f"q{i:02d}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"[{i}/{len(questions)}] {q!r} raw={rec['raw_chars']} compact={len(rec['compacted'])} "
            f"{rec['elapsed_ms']:.0f}ms -> {path.name}"
        )
        await asyncio.sleep(2)  # be polite to the shared prod server
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump bot-identical retrieval outputs.")
    ap.add_argument("questions", nargs="*", help="Extra questions to dump")
    ap.add_argument("--file", default=None, help="File with one question per line")
    ap.add_argument("--out", default="dumps", help="Output dir (default: dumps)")
    args = ap.parse_args()
    questions = list(DEFAULT_QUESTIONS) + list(args.questions)
    if args.file:
        questions += Path(args.file).read_text(encoding="utf-8").splitlines()
    return asyncio.run(main_async(questions, Path(args.out)))


if __name__ == "__main__":
    raise SystemExit(main())
