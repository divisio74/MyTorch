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

The pre-trained network uses ReLU in the hidden layers and a softmax output
with the cross-entropy loss. ReLU is cheap, does not saturate on positive
inputs, and works well with He initialisation, though a unit can stop
producing gradients when its input stays negative (leaky ReLU avoids this).
Sigmoid and tanh are bounded and saturate at both ends, which gives small
gradients in deep networks. The measured comparison of ReLU, leaky ReLU, tanh
and sigmoid is in `docs/benchmarks.md`, and the full list of choices is in the
README.

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

## Current state

The analyzer uses this package: `source/analyzer/pipeline.py` loads a network
with `Network.load`, encodes the positions (`source/analyzer/encoding.py`),
predicts with `predict_classes` in prediction mode, and trains with `fit` in
training mode. Networks are saved and loaded as JSON (`Network.save`,
`Network.load`) and can be generated with `my_torch_generator`.

`Dense` accepts `init_norm="he"` or `"xavier"` and scales the initial weights
accordingly.
