import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVICE = ROOT / "assets" / "disease_advice_zh.json"
CLASSES = ROOT / "splits_combined" / "class_to_idx.json"


def test_advice_covers_all_classes():
    advice = json.loads(ADVICE.read_text(encoding="utf-8"))
    classes = json.loads(CLASSES.read_text(encoding="utf-8"))
    assert set(advice.keys()) == set(classes.keys()), "advice must cover every class"
    for name, item in advice.items():
        assert isinstance(item.get("摘要"), str) and item["摘要"], name
        assert isinstance(item.get("防治措施"), list) and item["防治措施"], name
        assert isinstance(item.get("预防"), list) and item["预防"], name
        assert isinstance(item.get("来源"), str) and item["来源"], name
