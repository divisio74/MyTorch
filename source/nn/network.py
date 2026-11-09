import json
import numpy as np
from source.nn.layers import Dense, Dropout
from source.nn.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

FORMAT = "mytorch-chess-v1"
LAYERS = {cls.name: cls for cls in (Dense, Dropout, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax)}


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        a = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, grad):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def set_training(self, training):
        for layer in self.layers:
            layer.training = training

    def predict(self, x):
        self.set_training(False)
        return self.forward(x)

    def predict_classes(self, x):
        return np.argmax(self.predict(x), axis=1)

    def predict_mc(self, x, passes=20):
        self.set_training(True)
        total = sum(self.forward(x) for _ in range(passes))
        self.set_training(False)
        return total / passes

    def get_params(self):
        return [{k: v.copy() for k, v in layer.params.items()} for layer in self.layers]

    def set_params(self, params):
        for layer, saved in zip(self.layers, params):
            for k, v in saved.items():
                layer.params[k] = v.copy()

    def fit(self, X, y, epochs, loss, optimizer, batch_size=None, X_val=None, y_val=None, patience=None, l2=0.0):
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")
        size = batch_size or n
        history = {"train": [], "val": []}
        best = None
        best_val = np.inf
        wait = 0
        for epoch in range(epochs):
            self.set_training(True)
            order = np.random.permutation(n)
            total = 0
            for start in range(0, n, size):
                idx = order[start:start + size]
                pred = self.forward(X[idx])
                total += loss.forward(pred, y[idx]) * len(idx)
                self.backward(loss.backward())
                if l2:
                    for layer in self.layers:
                        if "w" in layer.grads:
                            layer.grads["w"] = layer.grads["w"] + l2 * layer.params["w"]
                optimizer.step(self.layers)
            history["train"].append(total / n)
            if X_val is None:
                continue
            val = loss.forward(self.predict(X_val), y_val)
            history["val"].append(val)
            if val < best_val:
                best_val = val
                best = self.get_params()
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    break
        if best is not None and patience:
            self.set_params(best)
        self.set_training(False)
        return history

    def save(self, path):
        layers = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                layers.append({"type": layer.name,
                               "weights": layer.params["w"].tolist(),
                               "biases": layer.params["b"].tolist()})
            elif isinstance(layer, Dropout):
                layers.append({"type": layer.name, "rate": layer.rate})
            else:
                layers.append({"type": layer.name})
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"format": FORMAT, "layers": layers}, file)

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except OSError as err:
            raise ValueError(f"{path}: {err.strerror}") from None
        except ValueError:
            raise ValueError(f"{path}: not a valid neural network file") from None
        if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("layers"), list):
            raise ValueError(f"{path}: unknown neural network format")
        layers = []
        for item in data["layers"]:
            kind = item.get("type", "dense" if "weights" in item else None)
            if kind not in LAYERS:
                raise ValueError(f"{path}: unknown layer type '{kind}'")
            if kind == "dense":
                try:
                    w = np.array(item["weights"], dtype=float)
                    b = np.array(item["biases"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"{path}: invalid dense layer") from None
                if w.ndim != 2 or b.shape != (w.shape[1],):
                    raise ValueError(f"{path}: invalid dense layer shapes")
                layer = Dense(w.shape[0], w.shape[1])
                layer.params["w"] = w
                layer.params["b"] = b
                layer.grads["w"] = np.zeros_like(w)
                layer.grads["b"] = np.zeros_like(b)
            elif kind == "dropout":
                layer = Dropout(item.get("rate", 0.5))
            else:
                layer = LAYERS[kind]()
            layers.append(layer)
        return cls(layers)
