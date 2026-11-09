import numpy as np
from source.nn.layers import Layer


class ReLU(Layer):
    name = "relu"

    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_out):
        return (self.x > 0) * grad_out


class LeakyReLU(Layer):
    name = "leaky_relu"

    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope
        self.x = None

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, grad_out):
        return np.where(self.x > 0, 1.0, self.slope) * grad_out


class Sigmoid(Layer):
    name = "sigmoid"

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.s = s
        return s

    def backward(self, grad_out):
        return self.s * (1 - self.s) * grad_out


class Tanh(Layer):
    name = "tanh"

    def __init__(self):
        super().__init__()
        self.t = None

    def forward(self, x):
        self.t = np.tanh(x)
        return self.t

    def backward(self, grad_out):
        return (1 - self.t ** 2) * grad_out


class Softmax(Layer):
    name = "softmax"

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.s = e / np.sum(e, axis=1, keepdims=True)
        return self.s

    def backward(self, grad_out):
        dot = np.sum(grad_out * self.s, axis=1, keepdims=True)
        return self.s * (grad_out - dot)
