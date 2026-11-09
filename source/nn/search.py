import numpy as np
from source.nn.generator import build_network
from source.nn.losses import CrossEntropy
from source.nn.optimizers import Adam
from source.nn.data import accuracy

SPACE = {
    "lr": (0.0003, 0.01),
    "hidden": [[32], [64], [128], [64, 32], [128, 64]],
    "batch_size": [32, 64, 128, 256],
    "activation": ["relu", "leaky_relu", "tanh"],
    "dropout": [0.0, 0.1, 0.3],
    "l2": [0.0, 0.0001, 0.001],
}


def sample(rng, space):
    low, high = space["lr"]
    return {
        "lr": float(np.exp(rng.uniform(np.log(low), np.log(high)))),
        "hidden": space["hidden"][rng.randint(len(space["hidden"]))],
        "batch_size": space["batch_size"][rng.randint(len(space["batch_size"]))],
        "activation": space["activation"][rng.randint(len(space["activation"]))],
        "dropout": space["dropout"][rng.randint(len(space["dropout"]))],
        "l2": space["l2"][rng.randint(len(space["l2"]))],
    }


def random_search(X, y, X_val, y_val, trials=10, epochs=20, patience=3, seed=0, space=SPACE):
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(trials):
        params = sample(rng, space)
        config = dict(params, input=X.shape[1], output=y.shape[1])
        net = build_network(config)
        history = net.fit(X, y, epochs, CrossEntropy(), Adam(params["lr"]),
                          batch_size=params["batch_size"], X_val=X_val, y_val=y_val,
                          patience=patience, l2=params["l2"])
        results.append({"params": params,
                        "val_loss": min(history["val"]),
                        "val_accuracy": accuracy(net.predict(X_val), y_val),
                        "network": net})
    return sorted(results, key=lambda r: r["val_loss"])
