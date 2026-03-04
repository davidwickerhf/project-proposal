from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_group_ids: list[int]
    val_group_ids: list[int]
    test_group_ids: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_group_ids": self.train_group_ids,
            "val_group_ids": self.val_group_ids,
            "test_group_ids": self.test_group_ids,
        }


def generate_grouped_5fold_splits(
    group_ids: list[int],
    seed: int = 42,
    val_groups_per_fold: int = 50,
) -> list[FoldSplit]:
    if len(group_ids) != 500:
        raise ValueError(
            f"Expected exactly 500 groups for the locked design, got {len(group_ids)}."
        )

    shuffled = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    test_groups_per_fold = len(shuffled) // 5
    if test_groups_per_fold != 100:
        raise ValueError(
            "Grouped 5-fold design expects 100 test groups per fold (500 total groups)."
        )

    folds: list[FoldSplit] = []
    for fold in range(5):
        test_start = fold * test_groups_per_fold
        test_end = test_start + test_groups_per_fold
        test_ids = shuffled[test_start:test_end]
        test_set = set(test_ids)

        train_val = [g for g in shuffled if g not in test_set]
        if len(train_val) != 400:
            raise ValueError("Expected 400 train+val groups per fold.")

        val_start = fold * val_groups_per_fold
        val_ids = train_val[val_start : val_start + val_groups_per_fold]
        if len(val_ids) != 50:
            raise ValueError("Expected 50 validation groups per fold.")
        val_set = set(val_ids)
        train_ids = [g for g in train_val if g not in val_set]
        if len(train_ids) != 350:
            raise ValueError("Expected 350 training groups per fold.")

        folds.append(
            FoldSplit(
                fold=fold,
                train_group_ids=sorted(train_ids),
                val_group_ids=sorted(val_ids),
                test_group_ids=sorted(test_ids),
            )
        )
    return folds
