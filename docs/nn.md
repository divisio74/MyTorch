# Neural-network components

The reusable neural-network code is in `source/nn`. It is a small NumPy
implementation of a sequential feed-forward network.

## How it works

- `Dense` in `source/nn/layers.py` computes `x @ w + b` and its backward pass
  calculates gradients for the input, weights, and biases.
- Activation layers in `source/nn/activations.py` transform their input and
  implement the corresponding backward derivative:
  - `ReLU` returns `max(x, 0)`.
  - `Sigmoid` returns `1 / (1 + exp(-x))`.
  - `Tanh` returns `tanh(x)`.
- `MSE` in `source/nn/losses.py` measures mean squared error and supplies its
  gradient.
- `SGD` in `source/nn/optimizers.py` subtracts the learning-rate-scaled
  gradients from each layer's parameters.
- `Network` in `source/nn/network.py` calls each layer in order for a forward
  pass, reverses that order for backpropagation, and provides a basic `fit`
  loop.

The layers retain values from their most recent forward pass for use during
backpropagation. Consequently, a backward pass should follow the forward pass
whose gradients are being calculated.

## Activation choices

The analyzer's hidden activation is `tanh`: it is smooth, differentiable, and
keeps hidden values centered and bounded between -1 and 1. This suits its
signed piece-placement inputs, but can have small gradients for large
magnitudes. The analyzer currently has no output activation; it compares the
raw output scores to select a label.

The reusable `source/nn` package also provides ReLU and sigmoid. ReLU is a
simple hidden-layer choice that does not saturate on positive inputs, though
units can stop producing gradients when their input stays negative. Sigmoid is
bounded between 0 and 1 and can be useful for probability-like outputs, but
can saturate and yield small gradients at either extreme. These are
implementation tradeoffs, not benchmark conclusions: the repository does not
currently include a comparison of activation functions.

## Add a layer or activation

Implement a class following the existing `Layer` interface: provide
`forward(x)` and `backward(grad_out)`. Save any forward-pass value needed to
compute the derivative in `backward`. For example, an activation typically
saves its input or output. Then import and include the layer in the sequence
passed to `Network`:

```python
from source.nn.activations import ReLU
from source.nn.layers import Dense
from source.nn.network import Network

network = Network([
    Dense(n_input=64, n_output=16),
    ReLU(),
    Dense(n_input=16, n_output=8),
])
```

When adding a parameterized layer, initialize its `params` and matching `grads`
dictionaries. `SGD.step` updates each parameter using the gradient with the
same key. Add tests for output shape and numerical-gradient agreement; the
finite-difference checks in `tests/test_gradient.py` are an example.

## Current limitations

The analyzer builds a `Network` from the saved weight matrices, with a `Dense`,
`Tanh`, and `Dense` sequence. Prediction uses `Network.predict`; training uses
`Network.fit` with `MSE` and `SGD`. The analyzer continues to store weights in
its JSON model format and stores learned dense-layer biases when saving a
trained model. Older models without biases are loaded with zero biases.

Also, `Dense` accepts `init_norm="he"` or `"xavier"`, but although it computes
the corresponding scale, the current weight assignment uses the separate
`WDC` constant. Do not assume the selected initialization mode currently
changes the initialized weight scale. The analyzer replaces the initialized
weights and biases with values from its model before using the network.

`CrossEntropy` exists in `source/nn/losses.py`, but its `forward` and
`backward` methods are unfinished. The analyzer therefore trains with `MSE`.
