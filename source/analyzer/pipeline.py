import numpy as np
from source.analyzer.parser import parse_file
from source.analyzer.encoding import CLASSES, N_FEATURES, encode_many
from source.nn.network import Network
from source.nn.layers import Dense
from source.nn.losses import CrossEntropy
from source.nn.optimizers import Adam
from source.nn.data import one_hot, split, accuracy

CHUNK = 10000


def load_network(path):
    net = Network.load(path)
    dense = [layer for layer in net.layers if isinstance(layer, Dense)]
    if not dense:
        raise ValueError(f"{path}: network has no layer")
    n_input, n_output = dense[0].params["w"].shape[0], dense[-1].params["w"].shape[1]
    if n_input != N_FEATURES or n_output != len(CLASSES):
        raise ValueError(f"{path}: expected a network with {N_FEATURES} inputs and {len(CLASSES)} outputs, "
                         f"got {n_input} and {n_output}")
    return net


def run_predict(loadfile, chessfile):
    net = load_network(loadfile)
    entries = parse_file(chessfile)
    for start in range(0, len(entries), CHUNK):
        X = encode_many([entry.fen for entry in entries[start:start + CHUNK]])
        for index in net.predict_classes(X):
            print(CLASSES[index])


def run_train(loadfile, chessfile, savefile=None, val_ratio=0.1, seed=0, epochs=30, patience=5):
    net = load_network(loadfile)
    entries = parse_file(chessfile)
    for entry in entries:
        if entry.label not in CLASSES:
            raise ValueError(f"label '{entry.label}' is not handled by this network, expected one of {CLASSES}")
    X = encode_many([entry.fen for entry in entries])
    y = one_hot([CLASSES.index(entry.label) for entry in entries], len(CLASSES))
    X_train, y_train, X_val, y_val = split(X, y, val_ratio, seed)
    if len(X_train) == 0 or len(X_val) == 0:
        X_train, y_train, X_val, y_val = X, y, None, None
    np.random.seed(seed)
    history = net.fit(X_train, y_train, epochs, CrossEntropy(), Adam(0.001), batch_size=128,
                      X_val=X_val, y_val=y_val, patience=patience)
    net.save(savefile or loadfile)
    print(f"epochs: {len(history['train'])}")
    print(f"train accuracy: {accuracy(net.predict(X_train), y_train):.4f}")
    if X_val is not None:
        print(f"validation accuracy: {accuracy(net.predict(X_val), y_val):.4f}")
