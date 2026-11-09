import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from train import load_dataset
from source.nn.search import random_search


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    X, y, X_val, y_val, _, _ = load_dataset("data/dataset.npz", 24000)
    results = random_search(X, y, X_val, y_val, trials=trials, epochs=8, patience=3)
    summary = [{"params": r["params"], "val_loss": r["val_loss"], "val_accuracy": r["val_accuracy"]} for r in results]
    os.makedirs("logs", exist_ok=True)
    with open("logs/hyperparameter_search.json", "w") as file:
        json.dump(summary, file, indent=1)
    results[0]["network"].save("bonus/bonus_network_searched.nn")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
