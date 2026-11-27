import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
logs = root / "results" / "logs"
logs.mkdir(parents=True, exist_ok=True)

py = sys.executable
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"

SPLIT_DIR = "splits_combined"
MANIFEST = "splits_combined/plantdoc_test.csv"

exps = [
    ("e1", "cnn", ["--model", "cnn", "--no-augment"]),
    ("e2", "resnet18", ["--model", "resnet18", "--freeze", "--pretrained", "--no-augment"]),
    ("e3", "resnet18", ["--model", "resnet18", "--finetune-layer4", "--pretrained", "--no-augment"]),
    ("e4", "resnet18", ["--model", "resnet18", "--finetune-layer4", "--pretrained", "--augment"]),
]
seeds = [42, 43, 44]


def run(args, out_path, err_path):
    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        subprocess.run([py] + args, cwd=str(root), stdout=out, stderr=err, env=env)


(logs / "v2_started.txt").write_text("started\n", encoding="utf-8")
for seed in seeds:
    for exp, model, extra in exps:
        outdir_rel = f"results/combined132/seed{seed}/{exp}"
        outdir = root / outdir_rel
        val_metrics = outdir / "metrics" / f"{model}_metrics.json"
        ckpt = outdir / "checkpoints" / f"{model}_best.pt"
        ext_metrics = outdir / "metrics" / "external_metrics.json"

        if not (val_metrics.exists() and ckpt.exists()):
            train_args = ["scripts/train.py"] + extra + [
                "--device", "cuda", "--seed", str(seed),
                "--split-dir", SPLIT_DIR, "--output-dir", outdir_rel,
            ]
            run(train_args, logs / f"v2_s{seed}_{exp}.out", logs / f"v2_s{seed}_{exp}.err")

        if ckpt.exists() and not ext_metrics.exists():
            eval_args = [
                "scripts/evaluate_external.py",
                "--checkpoint", f"{outdir_rel}/checkpoints/{model}_best.pt",
                "--manifest", MANIFEST,
                "--device", "cuda", "--num-workers", "0",
                "--metrics-out", f"{outdir_rel}/metrics/external_metrics.json",
                "--predictions-out", f"{outdir_rel}/metrics/external_predictions.csv",
            ]
            run(eval_args, logs / f"v2_s{seed}_{exp}_ext.out", logs / f"v2_s{seed}_{exp}_ext.err")

(logs / "v2_all_done.txt").write_text("done\n", encoding="utf-8")
print("ALL DONE")
