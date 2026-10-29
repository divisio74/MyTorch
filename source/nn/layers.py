import numpy as np

Wdc = 0.01

class Layer:

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x): raise NotImplementedError

    def backward(self, grad_out): raise NotImplementedError


class Dense(Layer):

    def __init__(self, n_input, n_output):
        super().__init__()
        self.params["w"] = wdc * np.random.randn(n_input, n_output)
        self.params["b"] = np.zeros(n_output)
        self.grads["w"] = np.zeros_like(self.params["w"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.x = None

    def forward(self, x):
        self.x = x
        return (self.x @ self.params["w"]) + self.params["b"]

    def backward(self, grad_out):
        self.grads["b"] =  np.sum(grad_out, axis=0)
        self.grads["w"] =  self.x.T @ grad_out
        return grad_out @ self.params["w"].T

