import numpy as np


def one_hot(labels, n_classes):
    labels = np.asarray(labels, dtype=int)
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1
    return out


def split(X, y, ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    cut = int(len(X) * (1 - ratio))
    train, val = order[:cut], order[cut:]
    return X[train], y[train], X[val], y[val]


def accuracy(pred, target):
    return float(np.mean(np.argmax(pred, axis=1) == np.argmax(target, axis=1)))


def confusion_matrix(pred, target, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(np.argmax(pred, axis=1), np.argmax(target, axis=1)):
        matrix[t, p] += 1
    return matrix
