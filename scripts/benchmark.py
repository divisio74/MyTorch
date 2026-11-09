import json
import multiprocessing
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_dataset, run_training

BASE = {"hidden": [64, 32], "activation": "relu", "dropout": 0.0}
TRAIN = {"optimizer": "adam", "lr": 0.001, "batch_size": 128, "l2": 0.0}
DATA = {}


def variant(name, config=None, **train):
    return {"name": name, "config": dict(BASE, **(config or {})), "train": dict(TRAIN, **train)}


GROUPS = {
    "activation": [variant(a, {"activation": a}) for a in ("relu", "leaky_relu", "tanh", "sigmoid")],
    "optimizer": [variant("sgd lr=0.05", optimizer="sgd", lr=0.05),
                  variant("momentum lr=0.02", optimizer="momentum", lr=0.02),
                  variant("adam lr=0.001", optimizer="adam", lr=0.001)],
    "initialisation": [variant("he", {"init": "he"}), variant("xavier", {"init": "xavier"})],
    "architecture": [variant(str(h), {"hidden": h}) for h in ([32], [64], [128], [64, 32], [128, 64])],
    "batch_size": [variant(f"batch {b}", batch_size=b) for b in (32, 128, 512)],
    "learning_rate": [variant(f"lr {lr}", lr=lr) for lr in (0.0003, 0.001, 0.01)],
    "overfitting": [variant("no regularisation", patience=None),
                    variant("l2=0.001", patience=None, l2=0.001),
                    variant("dropout 0.3", {"dropout": 0.3}, patience=None),
                    variant("early stopping", patience=3),
                    variant("dropout 0.3 + l2 + early stopping", {"dropout": 0.3}, patience=3, l2=0.001)],
}
SETTINGS = {"overfitting": {"limit": 6000, "epochs": 25}}


def run(job):
    group, item = job
    setting = SETTINGS.get(group, {})
    data = DATA[setting.get("limit", 24000)]
    train = dict(item["train"])
    patience = train.pop("patience", 3)
    _, result = run_training(item["config"], data, seed=0, epochs=setting.get("epochs", 8), patience=patience, **train)
    return group, item["name"], result


def main():
    for limit in {s.get("limit", 24000) for s in list(SETTINGS.values()) + [{}]}:
        DATA[limit] = load_dataset("data/dataset.npz", limit)
    jobs = [(g, item) for g, items in GROUPS.items() for item in items]
    with multiprocessing.Pool(8) as pool:
        outputs = pool.map(run, jobs)
    results = {}
    for group, name, result in outputs:
        results.setdefault(group, {})[name] = result
    os.makedirs("docs/benchmarks", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    with open("logs/benchmark.json", "w") as file:
        json.dump(results, file)
    lines = ["# Benchmarks", ""]
    for group, runs in results.items():
        lines += [f"## {group}", "", "| variant | train acc | val acc | test acc | epochs | final train loss | best val loss | seconds |",
                  "|---|---|---|---|---|---|---|---|"]
        plt.figure(figsize=(6, 4))
        for name, r in runs.items():
            lines.append(f"| {name} | {r['train_accuracy']:.3f} | {r['val_accuracy']:.3f} | {r['test_accuracy']:.3f} | "
                         f"{r['epochs_run']} | {r['train_loss'][-1]:.3f} | {min(r['val_loss']):.3f} | {r['seconds']} |")
            plt.plot(r["val_loss"], label=name)
        plt.xlabel("epoch")
        plt.ylabel("validation loss")
        plt.title(group)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"docs/benchmarks/{group}.png", dpi=120)
        plt.close()
        lines += ["", f"![{group}](benchmarks/{group}.png)", ""]
    with open("docs/benchmarks_tables.md", "w") as file:
        file.write("\n".join(lines))
    print("done")


if __name__ == "__main__":
    main()
