# tests/test_rag_usage_stats.py
"""
Unit tests for brain.RagUsageStats and the Brain wrapper methods that
expose it (get_rag_usage_stats) plus grudge_level_for — used by the
/ragstats and /mood commands. Pure in-RAM logic, no network calls.
"""

from brain import Brain, RagUsageStats


def test_no_data_yet_gives_zero_hit_rate():
    stats = RagUsageStats()
    d = stats.as_dict()
    assert d == {
        "total_classified": 0,
        "needs_memory": 0,
        "facts_retrieved": 0,
        "hit_rate_pct": 0.0,
    }


def test_record_classification_counts_total_and_needs_memory():
    stats = RagUsageStats()
    stats.record_classification(needs_memory=False)
    stats.record_classification(needs_memory=True)
    stats.record_classification(needs_memory=True)

    d = stats.as_dict()
    assert d["total_classified"] == 3
    assert d["needs_memory"] == 2


def test_record_retrieval_only_counts_found_facts():
    stats = RagUsageStats()
    stats.record_retrieval(facts_found=True)
    stats.record_retrieval(facts_found=False)
    stats.record_retrieval(facts_found=True)

    assert stats.as_dict()["facts_retrieved"] == 2


def test_hit_rate_is_facts_retrieved_over_needs_memory():
    stats = RagUsageStats()
    for _ in range(4):
        stats.record_classification(needs_memory=True)
    stats.record_retrieval(facts_found=True)
    stats.record_retrieval(facts_found=True)
    stats.record_retrieval(facts_found=False)

    d = stats.as_dict()
    assert d["needs_memory"] == 4
    assert d["facts_retrieved"] == 2
    assert d["hit_rate_pct"] == 50.0


def test_brain_get_rag_usage_stats_reflects_internal_tracker(bot_config):
    brain = Brain(bot_config)
    assert brain.get_rag_usage_stats() == {
        "total_classified": 0,
        "needs_memory": 0,
        "facts_retrieved": 0,
        "hit_rate_pct": 0.0,
    }

    brain._rag_usage.record_classification(needs_memory=True)
    brain._rag_usage.record_retrieval(facts_found=True)

    d = brain.get_rag_usage_stats()
    assert d["needs_memory"] == 1
    assert d["facts_retrieved"] == 1
    assert d["hit_rate_pct"] == 100.0


def test_brain_grudge_level_for_wraps_grudge_tracker(bot_config):
    brain = Brain(bot_config)
    chat_id, user_id = 555, 777

    assert brain.grudge_level_for(chat_id, user_id) == 0

    brain._grudge.record_attack((chat_id, user_id))
    assert brain.grudge_level_for(chat_id, user_id) == 1

    # A different chat with the same user_id must not share the grudge.
    assert brain.grudge_level_for(999, user_id) == 0
