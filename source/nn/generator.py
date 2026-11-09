import json
import os
import sys
from source.nn.layers import Dense, Dropout
from source.nn.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from source.nn.network import Network

HIDDEN_ACTIVATIONS = {"relu": ReLU, "leaky_relu": LeakyReLU, "sigmoid": Sigmoid, "tanh": Tanh}

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file containing description of a neural network we want to generate.
    nb_i            Number of neural networks to generate based on the configuration file."""


def load_config(path):
    try:
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
    except OSError as err:
        raise ValueError(f"{path}: {err.strerror}") from None
    except ValueError:
        raise ValueError(f"{path}: invalid configuration file") from None
    if not isinstance(config, dict):
        raise ValueError(f"{path}: invalid configuration file")
    return config


def build_network(config):
    try:
        n_input = int(config["input"])
        n_output = int(config["output"])
        hidden = [int(h) for h in config.get("hidden", [])]
        dropout = float(config.get("dropout", 0.0))
    except (KeyError, TypeError, ValueError):
        raise ValueError("configuration needs integer 'input', 'output' and a list 'hidden'") from None
    activation = config.get("activation", "relu")
    if activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"unknown activation '{activation}'")
    if min([n_input, n_output] + hidden) < 1:
        raise ValueError("layer sizes must be positive")
    init = config.get("init", "he" if activation in ("relu", "leaky_relu") else "xavier")
    sizes = [n_input] + hidden + [n_output]
    layers = []
    for i in range(len(sizes) - 1):
        if i == len(sizes) - 2:
            layers.append(Dense(sizes[i], sizes[i + 1], "xavier"))
        else:
            layers.append(Dense(sizes[i], sizes[i + 1], init))
            layers.append(HIDDEN_ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(Dropout(dropout))
    layers.append(Softmax())
    return Network(layers)


def generate(args):
    if len(args) == 0 or len(args) % 2:
        raise ValueError("expected pairs of 'config_file nb'")
    for i in range(0, len(args), 2):
        config = load_config(args[i])
        if not args[i + 1].isdigit():
            raise ValueError(f"invalid number '{args[i + 1]}'")
        base = os.path.splitext(args[i])[0]
        for n in range(1, int(args[i + 1]) + 1):
            build_network(config).save(f"{base}_{n}.nn")


def main():
    args = sys.argv[1:]
    if args == ["--help"]:
        print(USAGE)
        return 0
    try:
        generate(args)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 84
    return 0
