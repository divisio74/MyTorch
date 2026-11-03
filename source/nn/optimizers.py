import numpy as np

class SGD:

    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for key in layer.params:
                layer.params[key] = layer.params[key] - self.lr * layer.grads[key]


