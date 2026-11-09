import numpy as np
from source.nn.network import Network
from source.nn.layers import Dense
from source.nn.activations import Sigmoid, ReLU
from source.nn.losses import MSE
from source.nn.optimizers import SGD, Adam

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def run(layers, optimizer, epochs):
    net = Network(layers)
    history = net.fit(X, y, epochs, MSE(), optimizer)
    assert history["train"][-1] < 0.01
    assert (net.predict(X).round().ravel() == y.ravel()).all()


def test_sgd_sigmoid():
    np.random.seed(0)
    run([Dense(2, 4, "xavier"), Sigmoid(), Dense(4, 1, "xavier"), Sigmoid()], SGD(0.5), 5000)


def test_adam_relu():
    np.random.seed(0)
    run([Dense(2, 8, "he"), ReLU(), Dense(8, 1, "xavier"), Sigmoid()], Adam(0.05), 1000)


if __name__ == "__main__":
    test_sgd_sigmoid()
    test_adam_relu()
    print("OK")
