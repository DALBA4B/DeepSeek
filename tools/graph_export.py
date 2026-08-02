# graph_export.py
"""
Export the LightRAG knowledge graph to GEXF for viewing in Gephi.

Why this exists
---------------
LightRAG's built-in graph view has one force-directed layout, no filtering, and
a server-side cap of MAX_GRAPH_NODES (1000 here) on the whole-graph endpoint —
so most of the graph is never drawn. Gephi has none of those limits: filter
edges by weight to strip one-off mentions, colour by detected community, and
pick from a dozen layouts.

The node cap is worked around by fetching per-label subgraphs (`label=<name>`
is not capped) and merging them, rather than asking for the whole graph at once.

What ends up in the file
------------------------
Nodes carry `label`, `entity_type`, `description`, `degree` (number of
connections) and `file_path` (which day's document introduced them). Edges carry
`weight` and `description`. Gephi filters and styles on any of these.

Usage
-----
    python tools/graph_export.py                  # -> graph.gexf
    python tools/graph_export.py --out my.gexf
    python tools/graph_export.py --min-weight 2   # drop one-off connections upfront
"""

import argparse
import asyncio
import html
import logging
import sys
import urllib.parse
from typing import Dict, List, Optional

import aiohttp

from pathlib import Path

# This tool lives in tools/, one level below the bot's modules. Put the project
# root on sys.path so it runs the same from anywhere: python tools/graph_export.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from models import BotConfig
from rag_client import RagClient, RagClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("graph_export")

_OUT_DEFAULT = "graph.gexf"
_CONCURRENCY = 8


async def _get(rag: RagClient, path: str, params: dict, timeout: float) -> dict:
    token = await rag._ensure_token()
    url = f"{rag.base_url}{path}?{urllib.parse.urlencode(params)}"
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RagClientError(f"GET {path} -> {resp.status}: {text[:300]}")
            return await resp.json(content_type=None)


async def fetch_labels(rag: RagClient) -> List[str]:
    data = await _get(rag, "/graph/label/list", {}, timeout=180)
    labels = data if isinstance(data, list) else data.get("labels") or data.get("data") or []
    return [str(x) for x in labels]


async def fetch_full_graph(rag: RagClient, labels: List[str]) -> tuple:
    """
    Merge per-label subgraphs into one graph.

    The whole-graph query (`label="*"`) is capped server-side by
    MAX_GRAPH_NODES; per-label queries are not, so walking the label list and
    merging gets everything without touching the server config.
    """
    nodes: Dict[str, dict] = {}
    edges: Dict[str, dict] = {}
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    done = 0

    async def one(label: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                data = await _get(
                    rag, "/graphs",
                    {"label": label, "max_depth": 1, "max_nodes": 1000},
                    timeout=120,
                )
            except RagClientError as e:
                logger.warning("  %s: %s", label, e)
                return
        for node in data.get("nodes", []):
            nid = node.get("id")
            if nid and nid not in nodes:
                nodes[nid] = node
        for edge in data.get("edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            if not src or not tgt:
                continue
            key = tuple(sorted((src, tgt)))
            if key not in edges:
                edges[key] = edge
        done += 1
        if done % 200 == 0:
            logger.info("  %d/%d labels — %d nodes, %d edges so far",
                        done, len(labels), len(nodes), len(edges))

    await asyncio.gather(*(one(l) for l in labels))
    return nodes, edges


def _esc(value, limit: int = 0) -> str:
    """
    XML-escape a value, optionally truncating first.

    Truncation happens BEFORE escaping: cutting afterwards can slice through an
    entity like `&quot;` and leave `&qu`, which makes the whole file unparseable.
    Control characters are stripped for the same reason — LightRAG descriptions
    occasionally carry them.
    """
    text = str(value or "")
    if limit:
        text = text[:limit]
    text = "".join(ch for ch in text if ch >= " " or ch in "\t")
    return html.escape(text, quote=True)


def write_gexf(path: str, nodes: Dict[str, dict], edges: Dict[str, dict], min_weight: float) -> None:
    degree: Dict[str, int] = {nid: 0 for nid in nodes}
    kept = []
    for edge in edges.values():
        weight = float(edge.get("properties", {}).get("weight") or 1.0)
        if weight < min_weight:
            continue
        src, tgt = edge["source"], edge["target"]
        if src not in nodes or tgt not in nodes:
            continue
        degree[src] += 1
        degree[tgt] += 1
        kept.append((src, tgt, weight, edge))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n')
        f.write('<graph mode="static" defaultedgetype="undirected">\n')

        f.write('<attributes class="node">\n')
        f.write('  <attribute id="0" title="entity_type" type="string"/>\n')
        f.write('  <attribute id="1" title="description" type="string"/>\n')
        f.write('  <attribute id="2" title="degree" type="integer"/>\n')
        f.write('  <attribute id="3" title="source_doc" type="string"/>\n')
        f.write('</attributes>\n')
        f.write('<attributes class="edge">\n')
        f.write('  <attribute id="0" title="description" type="string"/>\n')
        f.write('</attributes>\n')

        f.write("<nodes>\n")
        for nid, node in nodes.items():
            props = node.get("properties", {}) or {}
            f.write(f'  <node id="{_esc(nid)}" label="{_esc(nid)}">\n')
            f.write('    <attvalues>\n')
            f.write(f'      <attvalue for="0" value="{_esc(props.get("entity_type"))}"/>\n')
            f.write(f'      <attvalue for="1" value="{_esc(props.get("description"), 900)}"/>\n')
            f.write(f'      <attvalue for="2" value="{degree.get(nid, 0)}"/>\n')
            f.write(f'      <attvalue for="3" value="{_esc(props.get("file_path"))}"/>\n')
            f.write('    </attvalues>\n')
            f.write('  </node>\n')
        f.write("</nodes>\n")

        f.write("<edges>\n")
        for i, (src, tgt, weight, edge) in enumerate(kept):
            desc = _esc((edge.get("properties", {}) or {}).get("description"), 500)
            f.write(f'  <edge id="{i}" source="{_esc(src)}" target="{_esc(tgt)}" weight="{weight}">\n')
            f.write(f'    <attvalues><attvalue for="0" value="{desc}"/></attvalues>\n')
            f.write('  </edge>\n')
        f.write("</edges>\n")
        f.write("</graph>\n</gexf>\n")

    isolated = sum(1 for nid in nodes if degree.get(nid, 0) == 0)
    logger.info("Wrote %s: %d nodes (%d with no connections), %d edges",
                path, len(nodes), isolated, len(kept))


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
    if rag is None:
        logger.error("LightRAG not configured (.env)")
        return 1
    if not await rag.health():
        logger.error("LightRAG health check failed")
        return 1

    labels = await fetch_labels(rag)
    logger.info("Graph has %d entities — fetching neighbourhoods...", len(labels))

    nodes, edges = await fetch_full_graph(rag, labels)
    logger.info("Merged: %d nodes, %d unique edges", len(nodes), len(edges))

    write_gexf(args.out, nodes, edges, args.min_weight)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the LightRAG graph to GEXF for Gephi.")
    parser.add_argument("--out", default=_OUT_DEFAULT, help=f"Output file (default {_OUT_DEFAULT})")
    parser.add_argument(
        "--min-weight", type=float, default=0.0, dest="min_weight",
        help="Drop edges below this weight (default 0 = keep all; filter in Gephi instead)",
    )
    args = parser.parse_args()

    config = load_config()
    sys.exit(asyncio.run(run(args, config)))


if __name__ == "__main__":
    main()
