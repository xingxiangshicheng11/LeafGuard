import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import analysis


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment metrics and draw figures")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3", "e4"])
    parser.add_argument("--figures", default="results/figures")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = analysis.collect(args.results_root, args.experiments)
    if not rows:
        print("No metrics found under", args.results_root)
        return

    print(f"{'exp':<5}{'model':<10}{'acc':>8}{'macroF1':>10}{'macroR':>10}{'train_s':>10}")
    for r in rows:
        print(f"{r['experiment']:<5}{r['model']:<10}{r['accuracy']:>8.4f}{r['macro_f1']:>10.4f}"
              f"{r['macro_recall']:>10.4f}{r['training_seconds']:>10.0f}")

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for r in rows:
        exp = r["experiment"]
        metrics_dir = Path(args.results_root) / exp / "metrics"
        metrics = analysis.load_json(metrics_dir / f"{r['model']}_metrics.json")
        analysis.plot_confusion(
            metrics.get("confusion_matrix"),
            metrics.get("class_names"),
            figures / f"{exp}_confusion.png",
            title=f"{exp} confusion ({r['model']})",
        )
        history_path = metrics_dir / f"{r['model']}_history.json"
        if history_path.exists():
            analysis.plot_history(
                analysis.load_json(history_path),
                figures / f"{exp}_curves.png",
                title=f"{exp} curves ({r['model']})",
            )
        print(f"\n[{exp}] top confusions:")
        for count, true_name, pred_name in analysis.top_confusions(
            metrics.get("confusion_matrix"), metrics.get("class_names"), args.top_k
        ):
            print(f"  {count:5d}  {true_name}  ->  {pred_name}")

    print(f"\nFigures written to {figures}")


if __name__ == "__main__":
    main()
