
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, grad):
        g = grad
        r_layer = reversed(self.layers)
        for layer in r_layer:
            g = layer.backward(g)
        return g

    def predict(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def fit(self, X, y, epochs, loss, optimizer):
        loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_value = loss.forward(y_pred, y)
            loss_grad = loss.backward()
            self.backward(loss_grad)
            optimizer.step(self.layers)
            loss_history.append(loss_value)
        return loss_history





