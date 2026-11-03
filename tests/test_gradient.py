import numpy as np
from source.nn.layers import Dense
from source.nn.activations import ReLU, Sigmoid
from source.nn.losses import MSE

EPS = 1e-5


def numerical_grad(f, arr):
    grad = np.zeros_like(arr)
    for i in np.ndindex(arr.shape):
        old = arr[i]
        arr[i] = old + EPS
        plus = f()
        arr[i] = old - EPS
        minus = f()
        arr[i] = old
        grad[i] = (plus - minus) / (2 * EPS)
    return grad


def test_dense():
    layer = Dense(3, 2)
    x = np.random.randn(5, 3)
    out = layer.forward(x)
    assert out.shape == (5, 2)
    grad_x = layer.backward(2 * out)

    def loss():
        return np.sum(layer.forward(x) ** 2)

    assert np.allclose(layer.grads["w"], numerical_grad(loss, layer.params["w"]))
    assert np.allclose(layer.grads["b"], numerical_grad(loss, layer.params["b"]))
    assert np.allclose(grad_x, numerical_grad(loss, x))


def test_activations():
    for cls in (ReLU, Sigmoid):
        layer = cls()
        x = np.random.randn(5, 3)
        out = layer.forward(x)
        grad_x = layer.backward(2 * out)
        assert np.allclose(grad_x, numerical_grad(lambda: np.sum(cls().forward(x) ** 2), x))


def test_mse():
    loss = MSE()
    pred = np.random.randn(5, 3)
    target = np.random.randn(5, 3)
    loss.forward(pred, target)
    assert np.allclose(loss.backward(), numerical_grad(lambda: MSE().forward(pred, target), pred))


if __name__ == "__main__":
    test_dense()
    test_activations()
    test_mse()
    print("OK")
