from __future__ import annotations

from .splits import DEFAULT_RATIOS


def partition_groups_by_class(groups, items):
    """Split global near-duplicate groups into per-class groups.

    A global group that spans multiple classes is not linked across classes;
    each class subset becomes its own group and the cross-class group is
    reported separately.
    """
    groups_by_class = {}
    cross_class_groups = []

    for group in groups:
        by_class = {}
        for index in group:
            item = items[index]
            by_class.setdefault(item["class_name"], []).append(item)
        if len(by_class) > 1:
            cross_class_groups.append(
                {
                    "size": len(group),
                    "classes": sorted(by_class),
                    "paths": [item["path"] for item in (items[i] for i in group)],
                }
            )
        for class_name, members in by_class.items():
            groups_by_class.setdefault(class_name, []).append(members)

    return groups_by_class, cross_class_groups


def _group_sort_key(group):
    return (-len(group), min(item["path"] for item in group))


def assign_groups(groups, ratios=None):
    """Assign whole groups to train/val/test to approximate image ratios."""
    ratios = ratios or DEFAULT_RATIOS
    total = sum(len(group) for group in groups)
    if total == 0:
        return {name: [] for name in ratios}

    targets = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: 0 for name in ratios}
    assignment = {name: [] for name in ratios}

    for group in sorted(groups, key=_group_sort_key):
        split_name = max(
            ratios,
            key=lambda name: (targets[name] - counts[name])
            / max(targets[name], 1e-9),
        )
        assignment[split_name].append(group)
        counts[split_name] += len(group)

    if len(groups) >= 3:
        for empty_split in ("val", "test"):
            if assignment[empty_split]:
                continue
            source = max(assignment, key=lambda name: len(assignment[name]))
            if len(assignment[source]) > 1:
                group = min(
                    assignment[source], key=lambda item: (len(item), _group_sort_key(item))
                )
                assignment[source].remove(group)
                assignment[empty_split].append(group)

    return assignment


def split_groups_by_class(groups_by_class, ratios=None, seed=42):
    """Create splits while keeping every near-duplicate group intact.

    ``seed`` is accepted for interface consistency; assignment itself is
    deterministic (groups sorted by size and path).
    """
    ratios = ratios or DEFAULT_RATIOS
    splits = {name: [] for name in ratios}

    for class_name in sorted(groups_by_class):
        assignment = assign_groups(groups_by_class[class_name], ratios)
        for split_name, groups in assignment.items():
            for group in groups:
                splits[split_name].extend(group)

    for split_name in splits:
        splits[split_name].sort(key=lambda item: (item["class_name"], item["path"]))
    return splits
