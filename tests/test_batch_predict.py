import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from scripts.batch_predict import collect_images, predict_batch, write_outputs
from src.models.cnn import SmallCNN


def make_image(path, color=(20, 120, 40)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_collect_images(tmp_path):
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "sub" / "b.png")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    recursive = collect_images(tmp_path)
    assert {path.name for path in recursive} == {"a.jpg", "b.png"}
    non_recursive = collect_images(tmp_path, recursive=False)
    assert {path.name for path in non_recursive} == {"a.jpg"}


def test_predict_batch_end_to_end(tmp_path):
    good_a = make_image(tmp_path / "images" / "a.png", (10, 100, 10))
    good_b = make_image(tmp_path / "images" / "b.png", (100, 10, 10))
    broken = tmp_path / "images" / "broken.png"
    broken.write_bytes(b"not an image")

    class_to_idx = {"ClassA": 0, "ClassB": 1}
    classes_path = tmp_path / "class_to_idx.json"
    classes_path.write_text(json.dumps(class_to_idx), encoding="utf-8")

    checkpoint = tmp_path / "ckpt.pth"
    torch.save(
        {
            "model_state_dict": SmallCNN(2).state_dict(),
            "num_classes": 2,
            "class_to_idx": class_to_idx,
            "model_name": "cnn",
            "input_size": 224,
            "config": {},
            "best_val_macro_f1": 0.0,
        },
        checkpoint,
    )

    rows, summary = predict_batch(
        [good_a, good_b, broken],
        checkpoint,
        classes_path=classes_path,
        device_arg="cpu",
        batch_size=2,
        num_workers=0,
        top_k=3,
    )
    assert summary["num_images"] == 3
    assert summary["num_ok"] == 2
    assert summary["num_failed"] == 1
    assert len(rows) == 3

    ok_rows = [row for row in rows if not row["error"]]
    assert len(ok_rows) == 2
    for row in ok_rows:
        assert row["pred_class"] in class_to_idx
        assert len(row["top_k_classes"].split("|")) == 2
        assert len(row["top_k_scores"].split("|")) == 2

    output_csv = tmp_path / "predictions.csv"
    summary_json = tmp_path / "summary.json"
    write_outputs(rows, summary, output_csv, summary_json)

    with open(output_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "path",
            "pred_class",
            "pred_label",
            "confidence",
            "top_k_classes",
            "top_k_scores",
            "error",
        ]
        assert len(list(reader)) == 3
    saved = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved["num_ok"] == 2
