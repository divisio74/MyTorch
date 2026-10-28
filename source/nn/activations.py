"""
    activation.py
    ...
"""

import numpy as np
from source.nn.layers import Layer


class ReLU(Layer):

    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_out):
        return (self.x > 0) * grad_out


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
        self.s = None

    def forward(self, x):
        s = 1 / (1 + np.exp(-x))
        self.s = s
        return s

    def backward(self, grad_out):
        return self.s * (1 - self.s) * grad_out


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.t = None

    def forward(self, x):
        self.t = np.tanh(x)
        return self.t

    def backward(self, grad_out):
        return (1 - self.t ** 2) * grad_out