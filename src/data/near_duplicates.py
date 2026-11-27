from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from PIL import Image


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def dhash(path, hash_size=8):
    """Difference hash as an integer plus its bit width."""
    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten().tolist():
        value = (value << 1) | int(bit)
    return value, hash_size * hash_size


def hamming_distance(left, right):
    return (left ^ right).bit_count()


def group_near_duplicates(records, max_distance=4, bands=None, bucket_limit=5000):
    """Group records whose dHash distance is <= max_distance.

    ``records`` must contain ``hash`` and ``bits`` keys. Uses banding so that
    any pair within the threshold shares at least one exact band (pigeonhole).
    """
    if not records:
        return [], {"skipped_buckets": 0}
    bits = records[0]["bits"]
    if any(record["bits"] != bits for record in records):
        raise ValueError("all records must use the same hash size")
    if bands is None:
        bands = max_distance + 1
    if bands < max_distance + 1:
        raise ValueError("bands must be >= max_distance + 1")
    chunk = (bits + bands - 1) // bands
    chunk_mask = (1 << chunk) - 1

    union_find = UnionFind(len(records))
    skipped_buckets = 0

    for band in range(bands):
        shift = band * chunk
        buckets = {}
        for index, record in enumerate(records):
            key = (record["hash"] >> shift) & chunk_mask
            buckets.setdefault(key, []).append(index)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            if len(bucket) > bucket_limit:
                skipped_buckets += 1
                continue
            for left, right in itertools.combinations(bucket, 2):
                if (
                    hamming_distance(records[left]["hash"], records[right]["hash"])
                    <= max_distance
                ):
                    union_find.union(left, right)

    groups = {}
    for index in range(len(records)):
        groups.setdefault(union_find.find(index), []).append(index)
    grouped = sorted(groups.values(), key=lambda group: (-len(group), group[0]))
    stats = {
        "skipped_buckets": skipped_buckets,
        "num_groups": len(grouped),
        "num_multi_groups": sum(1 for group in grouped if len(group) > 1),
        "num_grouped_images": sum(len(group) for group in grouped if len(group) > 1),
        "largest_group": max((len(group) for group in grouped), default=0),
    }
    return grouped, stats
