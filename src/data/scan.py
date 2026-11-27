from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve(path, project_root):
    path = Path(path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _relpath(path, project_root):
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, project_root)).as_posix()


def iter_class_dirs(data_root):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())


def iter_image_files(class_dir, extensions=IMAGE_EXTENSIONS):
    return sorted(
        p
        for p in Path(class_dir).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def is_readable_image(path, deep=True):
    try:
        with Image.open(path) as im:
            im.verify()
        if deep:
            with Image.open(path) as im:
                im.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(
    data_root,
    project_root=".",
    extensions=IMAGE_EXTENSIONS,
    hash_duplicates=False,
    include_images=False,
    deep=True,
):
    project_root = Path(project_root).resolve()
    data_root = _resolve(data_root, project_root)

    class_counts = {}
    damaged = []
    images = []
    num_files = 0

    for class_dir in iter_class_dirs(data_root):
        files = iter_image_files(class_dir, extensions)
        num_files += len(files)
        class_counts[class_dir.name] = 0
        for file_path in files:
            rel_path = _relpath(file_path, project_root)
            if not is_readable_image(file_path, deep=deep):
                damaged.append(rel_path)
                continue
            class_counts[class_dir.name] += 1
            images.append({"path": rel_path, "class_name": class_dir.name})

    duplicates = []
    cross_class_duplicates = []
    if hash_duplicates and images:
        by_hash = {}
        for item in images:
            digest = sha256_file(project_root / item["path"])
            by_hash.setdefault(digest, []).append(item)
        for group in by_hash.values():
            if len(group) < 2:
                continue
            paths = sorted(item["path"] for item in group)
            if len({item["class_name"] for item in group}) == 1:
                duplicates.append(paths)
            else:
                cross_class_duplicates.append(paths)
        duplicates.sort()
        cross_class_duplicates.sort()

    report = {
        "data_root": _relpath(data_root, project_root),
        "num_classes": len(class_counts),
        "num_files": num_files,
        "num_valid_images": len(images),
        "num_damaged": len(damaged),
        "classes": class_counts,
        "damaged": sorted(damaged),
        "duplicates": duplicates,
        "cross_class_duplicates": cross_class_duplicates,
    }
    if include_images:
        report["images"] = images
    return report
