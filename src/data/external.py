from __future__ import annotations


def map_external_items(items, class_map, class_to_idx):
    """Map external-dataset items onto the PlantVillage label space.

    Returns (mapped_items, skipped_counts). ``mapped_items`` contain
    ``path``, ``class_name`` (target class) and ``source_class``.
    Raises ValueError if a mapped target is absent from ``class_to_idx``.
    """
    mapped = []
    skipped = {}
    missing_targets = {}

    for item in items:
        source_class = item["class_name"]
        target_class = class_map.get(source_class)
        if target_class is None:
            skipped[source_class] = skipped.get(source_class, 0) + 1
            continue
        if target_class not in class_to_idx:
            missing_targets.setdefault(target_class, []).append(item["path"])
            continue
        mapped.append(
            {
                "path": item["path"],
                "class_name": target_class,
                "source_class": source_class,
            }
        )

    if missing_targets:
        raise ValueError(
            "mapped target classes missing from class_to_idx: "
            + ", ".join(sorted(missing_targets))
        )

    mapped.sort(key=lambda item: (item["class_name"], item["path"]))
    return mapped, skipped


def external_summary(mapped, skipped, damaged_count=0, duplicate_count=0):
    per_target = {}
    per_source = {}
    for item in mapped:
        per_target[item["class_name"]] = per_target.get(item["class_name"], 0) + 1
        per_source[item["source_class"]] = per_source.get(item["source_class"], 0) + 1
    return {
        "num_items": len(mapped),
        "num_target_classes": len(per_target),
        "per_target_class": dict(sorted(per_target.items())),
        "per_source_class": dict(sorted(per_source.items())),
        "skipped_unmapped": dict(sorted(skipped.items())),
        "damaged_excluded": damaged_count,
        "duplicate_copies_excluded": duplicate_count,
    }
