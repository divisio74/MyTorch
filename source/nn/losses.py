import numpy as np

EPS = 1e-12


class Loss:
    def forward(self, pred, target): raise NotImplementedError

    def backward(self): raise NotImplementedError


class MSE(Loss):
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size


class CrossEntropy(Loss):
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return -np.sum(target * np.log(np.clip(pred, EPS, 1))) / pred.shape[0]

    def backward(self):
        return -self.target / np.clip(self.pred, EPS, 1) / self.pred.shape[0]
