import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from scripts.evaluate_external import evaluate_manifest, write_outputs
from src.data.evaluation import compute_external_metrics


def make_image(path, color=(20, 120, 40)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_compute_external_metrics_known_values():
    metrics = compute_external_metrics(
        [0, 0, 1, 1], [0, 1, 1, 2], labels=[0, 1]
    )
    assert metrics["num_samples"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["out_of_label_predictions"] == 1
    assert metrics["confusion_matrix"] == [[1, 1], [0, 1]]
    assert metrics["per_class"][0]["recall"] == pytest.approx(0.5)


def test_evaluate_manifest_end_to_end(tmp_path):
    root = tmp_path / "data/plantdoc"
    make_image(root / "ClassA" / "a1.png", (10, 100, 10))
    make_image(root / "ClassA" / "a2.png", (20, 100, 10))
    make_image(root / "ClassB" / "b1.png", (100, 10, 10))
    make_image(root / "ClassB" / "b2.png", (110, 10, 10))

    manifest = tmp_path / "plantdoc_eval.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name"])
        for class_name, label in (("ClassA", 0), ("ClassB", 1)):
            for index in (1, 2):
                writer.writerow(
                    [f"data/plantdoc/{class_name}/{class_name[-1].lower()}{index}.png", label, class_name]
                )

    class_to_idx = {"ClassA": 0, "ClassB": 1}
    classes_path = tmp_path / "class_to_idx.json"
    classes_path.write_text(json.dumps(class_to_idx), encoding="utf-8")

    sys_path = str(Path(__file__).resolve().parents[1])
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from src.models.cnn import SmallCNN

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

    result, rows = evaluate_manifest(
        checkpoint,
        manifest,
        classes_path=classes_path,
        project_root=tmp_path,
        device_arg="cpu",
        batch_size=2,
        num_workers=0,
    )
    assert result["num_samples"] == 4
    assert result["num_evaluated_classes"] == 2
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["class_names"] == ["ClassA", "ClassB"]
    assert len(rows) == 4
    assert {row["true_class"] for row in rows} == {"ClassA", "ClassB"}
    assert all("confidence" in row for row in rows)

    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    write_outputs(result, rows, metrics_path, predictions_path)

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["num_samples"] == 4
    with open(predictions_path, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 4
