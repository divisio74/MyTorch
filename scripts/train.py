import argparse
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.analyzer.encoding import CLASSES, N_FEATURES
from source.nn.generator import build_network
from source.nn.losses import CrossEntropy
from source.nn.optimizers import SGD, Momentum, Adam
from source.nn.data import one_hot, accuracy, confusion_matrix

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


def load_dataset(path, limit=None):
    data = np.load(path)
    X, y = data["X"], data["y"]
    if limit:
        X, y = X[:limit], y[:limit]
    n = len(X)
    a, b = int(n * 0.8), int(n * 0.9)
    return (X[:a], one_hot(y[:a], len(CLASSES)),
            X[a:b], one_hot(y[a:b], len(CLASSES)),
            X[b:], one_hot(y[b:], len(CLASSES)))


def run_training(config, data, seed=0, epochs=30, patience=5, optimizer="adam", lr=0.001, batch_size=128, l2=0.0):
    X, y, X_val, y_val, X_test, y_test = data
    np.random.seed(seed)
    net = build_network(dict(config, input=N_FEATURES, output=len(CLASSES)))
    start = time.time()
    history = net.fit(X, y, epochs, CrossEntropy(), OPTIMIZERS[optimizer](lr), batch_size=batch_size,
                      X_val=X_val, y_val=y_val, patience=patience, l2=l2)
    result = {
        "config": config, "seed": seed, "optimizer": optimizer, "lr": lr, "batch_size": batch_size, "l2": l2,
        "epochs_run": len(history["train"]), "seconds": round(time.time() - start, 1),
        "train_loss": history["train"], "val_loss": history["val"],
        "train_accuracy": accuracy(net.predict(X[:20000]), y[:20000]),
        "val_accuracy": accuracy(net.predict(X_val), y_val),
        "test_accuracy": accuracy(net.predict(X_test), y_test),
        "confusion_matrix": confusion_matrix(net.predict(X_test), y_test, len(CLASSES)).tolist(),
    }
    return net, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.npz")
    parser.add_argument("--out", default="my_torch_network_full.nn")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = {"hidden": args.hidden, "activation": args.activation, "dropout": args.dropout}
    net, result = run_training(config, load_dataset(args.data, args.limit), args.seed, args.epochs,
                               args.patience, args.optimizer, args.lr, args.batch_size, args.l2)
    net.save(args.out)
    os.makedirs("logs", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.out))[0]
    with open(f"logs/{name}_seed{args.seed}.json", "w") as file:
        json.dump(result, file, indent=1)
    os.makedirs("docs/benchmarks", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(result["train_loss"], label="train loss")
    plt.plot(result["val_loss"], label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"docs/benchmarks/{name}_curves.png", dpi=120)
    print(args.out, "val", result["val_accuracy"], "test", result["test_accuracy"], result["seconds"], "s")


if __name__ == "__main__":
    main()
