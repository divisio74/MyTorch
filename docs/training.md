# Training and reproducibility

Everything needed to regenerate the pre-trained network is in the repository.
Run the commands from the repository root.

## 1. Build the dataset

```sh
python3 scripts/build_dataset.py 50000 data/dataset.npz
```

Reads every labelled file in `assets/` (`check`, `checkmate`, `nothing`, each
split by piece count), keeps at most 50 000 positions per class with a fixed
seed, encodes them with `source/analyzer/encoding.py`, shuffles them and writes
`data/dataset.npz` (ignored by git because of its size).

## 2. Classes

`source/analyzer/encoding.py` defines the five classes, in this order:
`Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`.
The colour is the side giving the check. `assets/` has no stalemate position,
so stalemate is not handled.

## 3. Input encoding (971 features)

| features | meaning |
|---|---|
| 768 | one-hot piece on square (12 pieces x 64 squares) |
| 1 | white to move |
| 4 | castling rights |
| 64 | en passant target square |
| 128 | squares attacked by white / by black |
| 2 | white king in check / black king in check |
| 2 | number of checkers of each king (divided by 2) |
| 2 | free escape squares of each king (divided by 8) |

The attack features are computed from the board only, no label is used. On
18 000 positions taken from the assets, the check features agree with the
labels every time.

## 4. Train

```sh
python3 scripts/train.py --limit 200000 --hidden 128 64 --dropout 0.3 --batch-size 32 --l2 0.0001 --epochs 25 --patience 5 --out my_torch_network_full.nn
```

Options: `--hidden`, `--activation`, `--dropout`, `--optimizer`, `--lr`,
`--batch-size`, `--l2`, `--seed`. Each run writes `logs/<name>_seed<seed>.json`
with the configuration, the loss curves, the accuracies and the confusion
matrix, and `docs/benchmarks/<name>_curves.png` with the curves. The dataset is split 80 % train, 10 % validation, 10 % test.

## 5. Benchmarks

```sh
python3 scripts/benchmark.py
```

Writes `logs/benchmark.json`, `docs/benchmarks_tables.md` and the curves in
`docs/benchmarks/`. The analysis is in `docs/benchmarks.md`.

`scripts/final_selection.py` compares the six final candidate settings (the
result chose the configuration above; see `docs/chosen_config.md`).

## 6. Generate blank networks

```sh
./my_torch_generator configs/basic_network.conf 3
```

A configuration file is JSON: `input`, `output`, `hidden` (list of sizes),
`activation` (`relu`, `leaky_relu`, `tanh`, `sigmoid`), `dropout`, and
optionally `init` (`he` or `xavier`). The output layer is always followed by a
softmax.

## 7. Bonus

`bonus/hyperparameter_search.py` runs a random search over the learning rate,
the architecture, the batch size, the activation, the dropout and the L2 term.
