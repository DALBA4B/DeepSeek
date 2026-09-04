"""
Tests for compact_context() — the client-side trim in rag_client.py.

The trim exists because LightRAG spends its context budget on entity
descriptions before it ever reaches the chat excerpts, and this graph has hub
descriptions of 8-16k characters. We ask the server for a wide context and cut
it here instead.

The important cases are the failure ones: this runs on every message that needs
memory, and returning an empty or mangled context would silently make the bot
dumber rather than crash.
"""
import json


from rag_client import compact_context, _shrink_description


def _blob(entities, relations, chunks):
    """Build a context in the exact layout LightRAG returns."""
    def lines(rows):
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)

    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n"
        + lines(entities) + "\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n"
        + lines(relations) + "\n```\n\n"
        "Document Chunks (Each entry has a reference_id):\n\n```json\n"
        + lines(chunks) + "\n```\n"
    )


ENTS = [
    {"entity": "Максим Тян", "type": "person", "description": "A" * 5000},
    {"entity": "Tima", "type": "person", "description": "B" * 3000},
]
RELS = [{"entity1": "Максим Тян", "entity2": "Tima", "description": "спорят"}]
CHUNKS = [{"reference_id": str(i), "content": f"[2026-01-{i:02d}] сообщение {i}"}
          for i in range(1, 21)]


def test_descriptions_are_capped():
    out, stats = compact_context(_blob(ENTS, RELS, CHUNKS), 1200, 12)
    assert stats["recognised"]
    for ent in _parse_entities(out):
        assert len(ent["description"]) <= 1200


def test_chunks_are_limited_to_the_first_n():
    out, stats = compact_context(_blob(ENTS, RELS, CHUNKS), 1200, 12)
    assert stats["chunks_in"] == 20
    assert stats["chunks_kept"] == 12
    # Excerpts arrive ordered by relevance, so the prefix must be kept — not a
    # sample, not the tail.
    assert "сообщение 1" in out
    assert "сообщение 13" not in out
    assert "сообщение 20" not in out


def test_entities_and_relations_survive():
    """Trimming must not drop whole records — only shorten descriptions."""
    out, stats = compact_context(_blob(ENTS, RELS, CHUNKS), 1200, 12)
    assert stats["entities"] == 2
    names = {e["entity"] for e in _parse_entities(out)}
    assert names == {"Максим Тян", "Tima"}
    assert "спорят" in out


def test_it_actually_shrinks():
    out, stats = compact_context(_blob(ENTS, RELS, CHUNKS), 1200, 12)
    assert stats["chars_after"] < stats["chars_before"]
    assert stats["desc_chars_after"] < stats["desc_chars_before"]


def test_zero_settings_are_a_no_op():
    blob = _blob(ENTS, RELS, CHUNKS)
    out, stats = compact_context(blob, 0, 0)
    assert out == blob
    assert stats == {}


def test_unknown_layout_is_returned_untouched():
    """A LightRAG version change must degrade to old behaviour, not to nothing."""
    weird = "Some completely different context format\nwith no json fences"
    out, stats = compact_context(weird, 1200, 12)
    assert out == weird
    assert stats == {"recognised": False}


def test_empty_context():
    out, stats = compact_context("", 1200, 12)
    assert out == ""
    assert stats == {}


def test_malformed_entity_line_does_not_lose_the_context():
    blob = _blob(ENTS, RELS, CHUNKS).replace(
        '{"entity": "Tima"', '{"entity": BROKEN "Tima"', 1
    )
    out, stats = compact_context(blob, 1200, 12)
    # The broken record is dropped, but the good one and the excerpts remain.
    assert stats["recognised"]
    assert "Максим Тян" in out
    assert "сообщение 1" in out


def test_fewer_chunks_than_the_cap_are_all_kept():
    few = CHUNKS[:3]
    out, stats = compact_context(_blob(ENTS, RELS, few), 1200, 12)
    assert stats["chunks_kept"] == 3
    assert "сообщение 3" in out


class TestShrinkDescription:
    def test_exact_repeats_are_dropped(self):
        frag = ("Максим владеет айпадом. " * 20).strip()   # ~480 chars
        desc = f"{frag}<SEP>{frag}<SEP>другой факт"
        out = _shrink_description(desc, 2000)
        assert out.count(frag) == 1
        assert "другой факт" in out

    def test_longest_fragments_win(self):
        """Long fragments are LLM summaries; short ones are single-import scraps."""
        long_frag = "L" * 900
        out = _shrink_description(f"короткий скрап<SEP>{long_frag}", 1000)
        assert long_frag in out

    def test_single_oversized_fragment_is_truncated_not_dropped(self):
        out = _shrink_description("X" * 5000, 1200)
        assert len(out) == 1200

    def test_empty_description(self):
        assert _shrink_description("", 1200) == ""

    def test_cap_is_respected(self):
        desc = "<SEP>".join(f"факт номер {i} " * 30 for i in range(10))
        assert len(_shrink_description(desc, 800)) <= 800 + len("<SEP>") * 10


def _parse_entities(blob):
    body = blob.split("```json\n", 1)[1].split("```", 1)[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out
