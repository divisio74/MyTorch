import os
import subprocess
import sys
import tempfile
import numpy as np
from source.nn.network import Network
from source.nn.layers import Dense, Dropout
from source.nn.activations import ReLU, Softmax, Tanh
from source.nn.losses import CrossEntropy
from source.nn.optimizers import Adam, Momentum
from source.nn.generator import build_network
from source.nn.data import one_hot, split, accuracy
from source.nn.search import random_search


def toy_data(n=300):
    X = np.random.randn(n, 4)
    labels = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    return X, one_hot(labels, 3)


def test_training_classification():
    np.random.seed(0)
    X, y = toy_data()
    X_train, y_train, X_val, y_val = split(X, y, 0.2)
    net = build_network({"input": 4, "hidden": [16], "output": 3})
    history = net.fit(X_train, y_train, 100, CrossEntropy(), Adam(0.01), batch_size=32,
                      X_val=X_val, y_val=y_val, patience=10)
    assert history["train"][-1] < history["train"][0]
    assert accuracy(net.predict(X_val), y_val) > 0.85


def test_momentum():
    np.random.seed(0)
    X, y = toy_data()
    net = Network([Dense(4, 16), ReLU(), Dense(16, 3, "xavier"), Softmax()])
    history = net.fit(X, y, 50, CrossEntropy(), Momentum(0.05), batch_size=64)
    assert history["train"][-1] < history["train"][0]


def test_save_load():
    np.random.seed(0)
    net = Network([Dense(4, 8), Tanh(), Dropout(0.2), Dense(8, 3, "xavier"), Softmax()])
    X = np.random.randn(5, 4)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "net.nn")
        net.save(path)
        loaded = Network.load(path)
    assert np.allclose(net.predict(X), loaded.predict(X))


def test_load_errors():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "bad.nn")
        for content in ("", "not json", '{"format": "other", "layers": []}',
                        '{"format": "mytorch-chess-v1", "layers": [{"type": "nope"}]}'):
            with open(path, "w") as file:
                file.write(content)
            try:
                Network.load(path)
            except ValueError:
                continue
            raise AssertionError(content)
        try:
            Network.load(os.path.join(folder, "missing.nn"))
        except ValueError:
            return
        raise AssertionError("missing file")


def test_dropout_modes():
    np.random.seed(0)
    layer = Dropout(0.5)
    x = np.ones((100, 10))
    assert (layer.forward(x) == 0).any()
    layer.training = False
    assert (layer.forward(x) == x).all()


def test_mc_dropout():
    np.random.seed(0)
    net = Network([Dense(4, 8), ReLU(), Dropout(0.5), Dense(8, 3, "xavier"), Softmax()])
    probs = net.predict_mc(np.random.randn(5, 4), 10)
    assert np.allclose(probs.sum(axis=1), 1)


def test_generator_cli():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as folder:
        conf = os.path.join(folder, "small.conf")
        with open(conf, "w") as file:
            file.write('{"input": 6, "hidden": [4], "output": 3}')
        run = subprocess.run([sys.executable, os.path.join(root, "my_torch_generator"), conf, "2"],
                             capture_output=True, text=True)
        assert run.returncode == 0
        assert sorted(os.listdir(folder)) == ["small.conf", "small_1.nn", "small_2.nn"]
        bad = subprocess.run([sys.executable, os.path.join(root, "my_torch_generator"), conf, "x"],
                             capture_output=True, text=True)
        assert bad.returncode == 84 and bad.stderr


def test_random_search():
    np.random.seed(0)
    X, y = toy_data(200)
    X_train, y_train, X_val, y_val = split(X, y, 0.25)
    results = random_search(X_train, y_train, X_val, y_val, trials=3, epochs=5)
    assert len(results) == 3
    assert results[0]["val_loss"] <= results[-1]["val_loss"]


if __name__ == "__main__":
    test_training_classification()
    test_momentum()
    test_save_load()
    test_load_errors()
    test_dropout_modes()
    test_mc_dropout()
    test_generator_cli()
    test_random_search()
    print("OK")
