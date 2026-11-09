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

CONFIG = {"hidden": [128, 64], "activation": "relu"}
CANDIDATES = {
    "A: batch 128 (previous)": dict(batch_size=128, l2=0.0, dropout=0.0),
    "B: batch 32": dict(batch_size=32, l2=0.0, dropout=0.0),
    "C: batch 32 + l2 1e-4": dict(batch_size=32, l2=0.0001, dropout=0.0),
    "D: batch 32 + l2 1e-4 + dropout 0.1": dict(batch_size=32, l2=0.0001, dropout=0.1),
    "E: batch 32 + l2 1e-4 + dropout 0.3": dict(batch_size=32, l2=0.0001, dropout=0.3),
    "F: batch 64 + l2 1e-4": dict(batch_size=64, l2=0.0001, dropout=0.0),
}
DATA = {}


def run(item):
    name, params = item
    config = dict(CONFIG, dropout=params["dropout"])
    _, result = run_training(config, DATA["data"], seed=0, epochs=8, patience=3, optimizer="adam",
                             lr=0.001, batch_size=params["batch_size"], l2=params["l2"])
    return name, result


def main():
    DATA["data"] = load_dataset("data/dataset.npz", 24000)
    with multiprocessing.Pool(6) as pool:
        results = dict(pool.map(run, list(CANDIDATES.items())))
    with open("logs/final_selection.json", "w") as file:
        json.dump(results, file)
    plt.figure(figsize=(6.5, 4))
    for name, r in results.items():
        plt.plot(r["val_loss"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title("final selection (971-128-64-5, ReLU, Adam 0.001)")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("docs/benchmarks/final_selection.png", dpi=120)
    for name, r in results.items():
        print(f"{name} | val acc {r['val_accuracy']:.3f} | test acc {r['test_accuracy']:.3f} | "
              f"train acc {r['train_accuracy']:.3f} | best val loss {min(r['val_loss']):.3f} | {r['seconds']} s")


if __name__ == "__main__":
    main()
