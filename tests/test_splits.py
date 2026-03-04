from __future__ import annotations

from collections import Counter

import pytest

from src.evaluation.splits import generate_grouped_5fold_splits


def test_grouped_5fold_splits_counts_disjointness_and_coverage() -> None:
    group_ids = list(range(1, 501))
    folds = generate_grouped_5fold_splits(group_ids=group_ids, seed=42)

    assert len(folds) == 5

    test_membership = Counter()
    for fold in folds:
        train = set(fold.train_group_ids)
        val = set(fold.val_group_ids)
        test = set(fold.test_group_ids)

        assert len(train) == 350
        assert len(val) == 50
        assert len(test) == 100

        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        assert len(train | val | test) == 500
        for gid in test:
            test_membership[gid] += 1

    assert len(test_membership) == 500
    assert set(test_membership.values()) == {1}


def test_grouped_5fold_splits_are_deterministic_for_seed() -> None:
    group_ids = list(range(1, 501))
    a = generate_grouped_5fold_splits(group_ids=group_ids, seed=99)
    b = generate_grouped_5fold_splits(group_ids=group_ids, seed=99)
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]


def test_grouped_5fold_rejects_wrong_group_count() -> None:
    with pytest.raises(ValueError, match="Expected exactly 500 groups"):
        generate_grouped_5fold_splits(group_ids=list(range(1, 20)), seed=42)
