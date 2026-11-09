import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.analyzer.encoding import CLASSES, encode_many


def main():
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    out = sys.argv[2] if len(sys.argv) > 2 else "data/dataset.npz"
    rng = np.random.RandomState(0)
    fens = {label: [] for label in CLASSES}
    for path in sorted(glob.glob("assets/*/*.txt")):
        with open(path) as file:
            for line in file:
                tokens = line.split()
                if len(tokens) >= 7:
                    fens[" ".join(tokens[6:])].append(" ".join(tokens[:6]))
    X, y = [], []
    for index, label in enumerate(CLASSES):
        chosen = rng.permutation(len(fens[label]))[:per_class]
        X.append(encode_many([fens[label][i] for i in chosen]))
        y.append(np.full(len(chosen), index, dtype=np.int8))
        print(label, len(fens[label]), "->", len(chosen))
    X = np.concatenate(X)
    y = np.concatenate(y)
    order = rng.permutation(len(X))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, X=X[order], y=y[order])
    print("saved", out, X.shape)


if __name__ == "__main__":
    main()
