import numpy as np


class Layer:
    name = "layer"

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.training = True

    def forward(self, x): raise NotImplementedError

    def backward(self, grad_out): raise NotImplementedError


class Dense(Layer):
    name = "dense"

    def __init__(self, n_input, n_output, init_norm="he"):
        super().__init__()
        if init_norm == "xavier":
            std = np.sqrt(2 / (n_input + n_output))
        elif init_norm == "he":
            std = np.sqrt(2 / n_input)
        else:
            raise ValueError(f"unknown init '{init_norm}', expected 'xavier' or 'he'")
        self.params["w"] = std * np.random.randn(n_input, n_output)
        self.params["b"] = np.zeros(n_output)
        self.grads["w"] = np.zeros_like(self.params["w"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.x = None

    def forward(self, x):
        self.x = x
        return (self.x @ self.params["w"]) + self.params["b"]

    def backward(self, grad_out):
        self.grads["b"] = np.sum(grad_out, axis=0)
        self.grads["w"] = self.x.T @ grad_out
        return grad_out @ self.params["w"].T


class Dropout(Layer):
    name = "dropout"

    def __init__(self, rate=0.5):
        super().__init__()
        if not 0 <= rate < 1:
            raise ValueError("dropout rate must be in [0, 1)")
        self.rate = rate
        self.mask = None

    def forward(self, x):
        if not self.training or self.rate == 0:
            self.mask = None
            return x
        self.mask = (np.random.rand(*x.shape) >= self.rate) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask
