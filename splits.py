import random
from collections import defaultdict
from torch.utils.data import Subset


def stratified_video_split(dataset, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Splits an AADCDataset into train/val/test Subsets, stratified by category
    label so rare anomaly categories still appear in every split.

    Args:
        dataset: an AADCDataset instance (already has .labels populated)
        train_frac, val_frac, test_frac: must sum to 1.0
        seed: for reproducibility

    Returns:
        (train_subset, val_subset, test_subset) -- torch.utils.data.Subset objects
        wrapping the same underlying AADCDataset, so __getitem__ behavior
        (frame sampling, transforms) is unchanged.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, \
        "train/val/test fractions must sum to 1.0"

    rng = random.Random(seed)

    # Group video indices by category label
    by_label = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        by_label[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in by_label.items():
        indices = indices[:]
        rng.shuffle(indices)
        n = len(indices)

        if n < 3:
            cat_name = dataset.categories[label] if hasattr(dataset, "categories") else label
            print(f"WARNING: category '{cat_name}' has only {n} video(s) total -- "
                  f"it cannot be represented in all three splits. All {n} will go to train.")

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # whatever remains goes to test, guards against rounding shrinking it to 0
        n_test = n - n_train - n_val

        # If a category is too small to cover all 3 splits, guarantee at least
        # one video per split where possible, borrowing from train.
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_test == 0:
                n_test = 1
                n_train -= 1
            n_train = max(n_train, 0)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    return train_subset, val_subset, test_subset


def print_split_summary(dataset, train_subset, val_subset, test_subset):
    """Prints a per-category breakdown of how many videos landed in each split."""
    def count_by_label(subset):
        counts = defaultdict(int)
        for idx in subset.indices:
            counts[dataset.labels[idx]] += 1
        return counts

    train_counts = count_by_label(train_subset)
    val_counts = count_by_label(val_subset)
    test_counts = count_by_label(test_subset)

    print(f"{'Category':<35}{'Train':>8}{'Val':>8}{'Test':>8}{'Total':>8}")
    for label, cat in enumerate(dataset.categories):
        tr, va, te = train_counts.get(label, 0), val_counts.get(label, 0), test_counts.get(label, 0)
        print(f"{cat:<35}{tr:>8}{va:>8}{te:>8}{tr+va+te:>8}")
    print(f"{'TOTAL':<35}{len(train_subset):>8}{len(val_subset):>8}{len(test_subset):>8}{len(dataset):>8}")
