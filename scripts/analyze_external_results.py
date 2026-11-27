import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import analysis


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze PlantDoc external-evaluation metrics")
    parser.add_argument("--results-root", default="results/combined132")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3", "e4"])
    parser.add_argument("--class-to-idx", default="splits_combined/class_to_idx.json")
    parser.add_argument("--figures", default="results/figures/combined132/plantdoc")
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    class_to_idx = load_json(args.class_to_idx)
    idx_to_name = {int(v): k for k, v in class_to_idx.items()}
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    for exp in args.experiments:
        accs = []
        for seed in args.seeds:
            p = Path(args.results_root) / f"seed{seed}" / exp / "metrics" / "external_metrics.json"
            if not p.exists():
                continue
            m = load_json(p)
            accs.append(m["accuracy"])
            labels = [int(x) for x in m["labels"]]
            names = [idx_to_name.get(l, str(l)) for l in labels]
            if seed == args.plot_seed:
                analysis.plot_confusion(
                    m["confusion_matrix"], names,
                    figures / f"plantdoc_{exp}_seed{seed}_confusion.png",
                    title=f"PlantDoc external confusion - {exp} (seed{seed})",
                )
            if seed == args.plot_seed:
                print(f"\n[{exp} seed{seed}] external top confusions:")
                for count, true_name, pred_name in analysis.top_confusions(
                    m["confusion_matrix"], names, args.top_k
                ):
                    print(f"  {count:4d}  {true_name}  ->  {pred_name}")
                # weakest classes by F1 with support >= 5
                weak = []
                for lbl, stats in m["per_class"].items():
                    key = str(lbl)
                    nm = idx_to_name.get(int(key), key) if key.lstrip("-").isdigit() else key
                    if stats["support"] >= 5:
                        weak.append((stats["f1"], stats["support"], nm))
                weak.sort()
                print(f"[{exp} seed{seed}] weakest classes (F1, support):")
                for f1, sup, nm in weak[: args.top_k]:
                    print(f"  F1={f1:.3f}  n={sup}  {nm}")
        if accs:
            mean = sum(accs) / len(accs)
            print(f"\n{exp}: external accuracy mean over {len(accs)} seeds = {mean:.4f}")

    print(f"\nfigures written to {figures}")


if __name__ == "__main__":
    main()
