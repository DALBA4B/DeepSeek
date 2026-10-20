"""
Tests for rerank.py — client-side v3 selection of retrieved facts.

The module decides WHICH facts (entities + chat excerpts) reach the prompt
within the same character budget the plain trim used. These are pure-function
tests: scoring, junk rules, the absence gate and the budget filler. The
bot-level integration lives in test_rag_context_compaction.py.
"""
from rerank import (
    chunk_junk,
    common_prefix,
    compute_idf,
    content_words,
    entity_name,
    is_absent,
    is_junk_entity,
    score_fact,
    select_facts,
    stem,
    translit,
)


class TestContentWords:
    def test_function_words_and_short_tokens_are_dropped(self):
        assert content_words("кто такой Максим") == {"максим"}

    def test_ukrainian_letters_are_not_split(self):
        # The old [а-яё] tokenizer split "Співак" into "сп"+"вак" and the
        # question stopped matching its own entity.
        assert content_words("кто такой Співак") == {"співак"}

    def test_yo_is_normalized(self):
        assert "ежик" in content_words("Ёжик в тумане")


class TestBridges:
    def test_stem_drops_trailing_char_only_for_long_words(self):
        assert stem("Тимур") == "тиму"
        assert stem("Рома") == "рома"

    def test_translit_bridges_russian_to_latin(self):
        assert translit("тима") == "tima"

    def test_common_prefix(self):
        assert common_prefix("тимур", "тима") == 3


class TestJunkRules:
    def test_dates_and_tiny_names_are_junk(self):
        assert is_junk_entity("2022-01-31")
        assert is_junk_entity("Ф")

    def test_measures_years_and_money_are_not_junk(self):
        # Real facts this graph keeps — "почти сироты" are cosmetic, not junk.
        assert not is_junk_entity("100 км")
        assert not is_junk_entity("183 см")
        assert not is_junk_entity("2016 год")
        assert not is_junk_entity("150 грн")

    def test_photo_dump_chunk_is_junk(self):
        assert chunk_junk(" ".join(["[фото]"] * 16))

    def test_short_excerpt_is_junk(self):
        assert chunk_junk("[фото] пятница")

    def test_normal_excerpt_is_not_junk(self):
        text = " ".join(f"слово{i}" for i in range(10))
        assert not chunk_junk(text)


class TestIdfAndScore:
    def test_idf_floor_keeps_matches_above_zero(self):
        facts = ["ENTITY Максим: Максим тут"] * 3
        idf = compute_idf(facts, {"максим"})
        assert idf["максим"] > 0

    def test_rare_words_weigh_more(self):
        facts = ["Максим бегает", "Максим бегает", "Максим бегает", "кальян стоит"]
        idf = compute_idf(facts, {"максим", "кальян"})
        assert idf["кальян"] > idf["максим"]

    def test_raw_match_is_marked_raw(self):
        text = "ENTITY Максим: играет в доту"
        sc, parts = score_fact(text, {"максим"}, compute_idf([text], {"максим"}))
        assert sc > 0 and "максим:raw" in parts

    def test_stem_bridge(self):
        text = "CHUNK: Максим говорил про кальян вчера"
        sc, parts = score_fact(text, {"кальяна"}, compute_idf([text], {"кальяна"}))
        assert sc > 0 and "кальяна:stem" in parts

    def test_translit_bridge(self):
        text = "ENTITY tima: играет в доту"
        sc, parts = score_fact(text, {"тима"}, compute_idf([text], {"тима"}))
        assert sc > 0 and "тима:tr" in parts


class TestAbsenceGate:
    def test_literal_match_is_not_absent(self):
        texts = ["ENTITY Тима: играет"]
        assert is_absent(texts, {"тима"}) is False

    def test_bridge_hit_still_counts_as_absent(self):
        # "Тимур" vs "Timur" is similarity, not knowledge: the gate must fire.
        texts = ["ENTITY Timur: участник чата"]
        cw = {"тимур"}
        sc, _ = score_fact(texts[0], cw, compute_idf(texts, cw))
        assert sc > 0           # the translit bridge found it
        assert is_absent(texts, cw)

    def test_no_match_anywhere_is_absent(self):
        texts = ["ENTITY Максим: играет в доту"]
        assert is_absent(texts, {"вертолет"}) is True


class TestSelectFacts:
    def test_budget_is_respected(self):
        facts = [("chunk", i, None, f"CHUNK: {i} " + "слово " * 30) for i in range(10)]
        kept, stats = select_facts(facts, "слово", budget=400)
        assert kept
        assert stats["used"] <= 400
        assert stats["kept"] < 10

    def test_zero_score_facts_fill_the_tail(self):
        facts = [
            ("chunk", 0, None, "CHUNK: Максим говорил про кальян"),
            ("chunk", 1, None, "CHUNK: совсем другая болтовня тут была"),
        ]
        kept, stats = select_facts(facts, "кальян у максима", budget=10_000)
        assert stats["kept"] == 2      # nothing thrown away while budget allows
        assert [i for _, i, _, _ in kept] == [0, 1]

    def test_demoted_junk_loses_to_clean_fact_with_same_match(self):
        facts = [
            ("chunk", 0, None, "CHUNK: [фото] рыбалка"),
            ("chunk", 1, None, "CHUNK: большой разговор про рыбалку и удочки вчера вечером"),
        ]
        kept, _ = select_facts(facts, "рыбалка", budget=10_000)
        assert [i for _, i, _, _ in kept] == [1, 0]

    def test_date_named_entity_is_demoted(self):
        facts = [
            ("entity", 0, "2022-01-31", "ENTITY 2022-01-31: рыбалка на реке"),
            ("entity", 1, "Отчёт", "ENTITY Отчёт: рыбалка на реке"),
        ]
        kept, _ = select_facts(facts, "рыбалка", budget=10_000)
        assert [i for _, i, _, _ in kept] == [1, 0]

    def test_ties_keep_server_order(self):
        same = "CHUNK: раз два три четыре пять шесть семь восемь раз"
        facts = [("chunk", 0, None, same), ("chunk", 1, None, same)]
        kept, _ = select_facts(facts, "раз", budget=10_000)
        assert [i for _, i, _, _ in kept] == [0, 1]

    def test_absent_flag_in_stats(self):
        facts = [("entity", 0, "Максим", "ENTITY Максим: играет в доту")]
        _, stats = select_facts(facts, "вертолет", budget=1000)
        assert stats["absent"] is True

    def test_stats_counts(self):
        facts = [
            ("entity", 0, "Максим", "ENTITY Максим: играет"),
            ("chunk", 0, None, "CHUNK: играет в доту вместе с друзьями вчера"),
        ]
        kept, stats = select_facts(facts, "играет", budget=10_000)
        assert stats["facts"] == 2
        assert stats["kept"] == len(kept) == 2
        assert stats["kept_entities"] == 1
        assert stats["kept_chunks"] == 1


def test_entity_name_variants():
    assert entity_name({"entity": "Тима"}) == "Тима"
    assert entity_name({"name": "Тима"}) == "Тима"
    assert entity_name({}) == "?"
