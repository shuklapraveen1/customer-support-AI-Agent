import pytest

from src.data.split import assignment_lookup, split_conversations


def test_split_covers_all_ids_with_no_overlap():
    ids = [f"c{i}" for i in range(100)]
    assignment = split_conversations(ids, seed=42)

    all_assigned = assignment.train + assignment.dev + assignment.test
    assert sorted(all_assigned) == sorted(ids)
    assert set(assignment.train) & set(assignment.dev) == set()
    assert set(assignment.train) & set(assignment.test) == set()
    assert set(assignment.dev) & set(assignment.test) == set()


def test_split_ratios_are_approximately_correct_at_scale():
    ids = [f"c{i}" for i in range(1000)]
    assignment = split_conversations(ids, seed=42)
    n = len(ids)
    assert abs(len(assignment.train) / n - 0.7) < 0.02
    assert abs(len(assignment.dev) / n - 0.15) < 0.02
    assert abs(len(assignment.test) / n - 0.15) < 0.02


def test_split_is_deterministic_given_same_seed():
    ids = [f"c{i}" for i in range(200)]
    a1 = split_conversations(ids, seed=42)
    a2 = split_conversations(ids, seed=42)
    assert a1.train == a2.train
    assert a1.dev == a2.dev
    assert a1.test == a2.test


def test_split_result_independent_of_input_order():
    ids = [f"c{i}" for i in range(200)]
    shuffled = list(reversed(ids))
    a1 = split_conversations(ids, seed=42)
    a2 = split_conversations(shuffled, seed=42)
    assert a1.train == a2.train
    assert a1.dev == a2.dev
    assert a1.test == a2.test


def test_split_different_seed_gives_different_assignment():
    ids = [f"c{i}" for i in range(200)]
    a1 = split_conversations(ids, seed=42)
    a2 = split_conversations(ids, seed=7)
    assert a1.train != a2.train


def test_split_deduplicates_repeated_ids():
    ids = ["a", "a", "b", "c", "c", "c"]
    assignment = split_conversations(ids, seed=42)
    all_assigned = assignment.train + assignment.dev + assignment.test
    assert sorted(all_assigned) == ["a", "b", "c"]


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        split_conversations(["a", "b"], train=0.5, dev=0.3, test=0.3)


def test_no_conversation_is_dropped_under_rounding():
    for n in range(1, 30):
        ids = [f"c{i}" for i in range(n)]
        assignment = split_conversations(ids, seed=42)
        assert len(assignment.train) + len(assignment.dev) + len(assignment.test) == n


def test_assignment_lookup_covers_every_id_with_a_valid_label():
    ids = [f"c{i}" for i in range(10)]
    assignment = split_conversations(ids, seed=42)
    lookup = assignment_lookup(assignment)
    assert set(lookup.keys()) == set(ids)
    assert all(v in ("train", "dev", "test") for v in lookup.values())
