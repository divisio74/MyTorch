import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for key in layer.params:
                layer.params[key] -= self.lr * layer.grads[key]


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            for key in layer.params:
                v = self.velocity.get((i, key), np.zeros_like(layer.params[key]))
                v = self.beta * v - self.lr * layer.grads[key]
                self.velocity[(i, key)] = v
                layer.params[key] += v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            for key in layer.params:
                g = layer.grads[key]
                m = self.beta1 * self.m.get((i, key), 0) + (1 - self.beta1) * g
                v = self.beta2 * self.v.get((i, key), 0) + (1 - self.beta2) * g ** 2
                self.m[(i, key)] = m
                self.v[(i, key)] = v
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            if "w" in layer.params:
                layer.params["w"] *= 1 - self.lr * self.weight_decay
        super().step(layers)
