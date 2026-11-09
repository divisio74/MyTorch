# MyTorch chess analyzer


my torch networkd 

car find . -type  f -name "my_torch_netxwork* ne doit matcher qu'un model

A small NumPy-based neural-network project with a command-line analyzer for chess
positions written in FEN notation.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt` (`numpy` and `matplotlib`)

Install the dependencies from the repository root:

```sh
# Create a virtual environment if you don't already have one
python -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick start

```sh
python3 scripts/build_dataset.py 50000 data/dataset.npz   # encode and balance assets/
python3 scripts/train.py --limit 200000 --hidden 128 64 --dropout 0.3 --batch-size 32 --l2 0.0001 --epochs 25 --patience 5 --out my_torch_network_full.nn
python3 my_torch_analyzer --predict my_torch_network_full.nn positions.txt
```

The pre-trained network `my_torch_network_full.nn` is in the repository.

## Run the analyzer

The command-line entry point is `my_torch_analyzer`:

```sh
python3 my_torch_analyzer --help
python3 my_torch_analyzer --predict my_torch_network_full.nn positions.txt
python3 my_torch_analyzer --train my_torch_network_full.nn labeled-positions.txt
python3 my_torch_analyzer --train --save trained.nn my_torch_network_full.nn labeled-positions.txt
```

Training without `--save` updates the model at `LOADFILE`. With `--save`, the
updated model is written to `SAVEFILE` and `LOADFILE` is left unchanged. Errors
are written on the error output and the exit code is 84 (0 otherwise).

Each non-empty line in a chess input file starts with a six-field FEN position.
Training lines must end with a label; prediction labels are optional:

```text
4k3/8/8/8/8/8/8/4K3 w - - 0 1 Nothing
```

In prediction mode, one line is printed per board, in the same order, among
`Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`.
The colour is the side that gives the check. A network only accepts the
labels of its five classes: training with `Check`, `Checkmate` or `Stalemate`
is refused with an error (`assets/` has no stalemate position).

Generate blank networks from a configuration file:

```sh
./my_torch_generator configs/basic_network.conf 3   # basic_network_1.nn ... _3.nn
```

## Project layout

| path | content |
|---|---|
| `source/nn/` | neural-network module: layers, activations, losses, optimisers, `Network`, save/load, generator, random search |
| `source/analyzer/` | FEN parser, FEN encoding, CLI modes, glue between the CLI and the network (`pipeline.py`) |
| `scripts/` | dataset building, training, benchmarks (needed to rebuild the network) |
| `bonus/` | hyperparameter random search |
| `configs/` | example configuration for the generator |
| `logs/` | one JSON per training run (configuration, seed, curves, accuracies, confusion matrix) and the benchmark results |
| `docs/` | [chosen_config.md](docs/chosen_config.md), [training.md](docs/training.md), [benchmarks.md](docs/benchmarks.md), [benchmarks_tables.md](docs/benchmarks_tables.md), [nn.md](docs/nn.md), [analyser.md](docs/analyser.md), [test.md](docs/test.md) |

## The neural network

Network: `971 -> 128 -> 64 -> 5`, ReLU, He initialisation on hidden layers,
Xavier on the output layer, softmax, cross-entropy loss, Adam (0.001),
mini-batches of 32, dropout 0.3, L2 0.0001, early stopping. Each value is
justified, with the benchmark images, in [docs/chosen_config.md](docs/chosen_config.md). Pure NumPy, no machine-learning library. The `.nn` file is
JSON (`"format": "mytorch-chess-v1"`, list of layers with weights and biases).

The input has 971 features: piece-on-square one-hot (768), side to move,
castling rights, en passant square, squares attacked by each side (128), and
for each king: in check, number of checkers, free escape squares. These are
computed from the board only, never from the label. Details in
[docs/training.md](docs/training.md).

### Results

Pre-trained network, 200 000 balanced positions (160 000 train / 20 000
validation / 20 000 test):

| train | validation | test | 25 000 held-out positions |
|---|---|---|---|
| 96.5 % | 96.3 % | 96.4 % | 96.3 % |

`Nothing` and the colour of the side giving check are never mixed up. All
errors are between a check and a checkmate of the same colour (the network
estimates whether the king can run away, but does not simulate blocking or
capturing the checking piece). Training and validation accuracy are almost
equal: the network does not overfit. The earlier network (batch 128, no
regularisation, 150 000 positions) got 95.8 % on the same held-out positions;
it is kept in `bonus/bonus_network_batch128.nn`. Curves, confusion matrix and
details: [docs/chosen_config.md](docs/chosen_config.md).

## Design choices

Each choice, the reason, and the evidence. Measured results come from
`scripts/benchmark.py` (one variable changed at a time, 24 000 positions,
8 epochs, seed 0). Full tables: [docs/benchmarks_tables.md](docs/benchmarks_tables.md).

| choice | kept | tried / rejected | why |
|---|---|---|---|
| hidden activation | ReLU | leaky ReLU, tanh, sigmoid | ReLU, leaky ReLU and tanh are equivalent (test accuracy 0.920 / 0.921 / 0.920). Sigmoid is worse (0.875): its derivative is at most 0.25, the gradient shrinks layer after layer. ReLU is the cheapest and does not saturate. |
| output activation | softmax | sigmoid, none | five exclusive classes need probabilities that sum to 1; softmax with cross-entropy gives the simple gradient `(p - y) / N`. |
| loss | cross-entropy | MSE | MSE only used to validate the library on XOR; cross-entropy is the loss made for classification. |
| initialisation | He (hidden), Xavier (output) | Xavier, constant 0.01 | He 0.920 vs Xavier 0.917 with ReLU (inside the noise, but consistent with the theory: He compensates the half of the signal that ReLU removes). A constant scale of 0.01 blocked the XOR network at a loss of 0.25: the signal vanishes through the layers. |
| optimiser | Adam (0.001) | SGD, momentum | SGD at 0.05 is the slowest (0.881). Momentum 0.931 and Adam 0.920 are close; Adam needs far less learning-rate tuning. |
| learning rate | 0.001 | 0.0003, 0.01 | 0.0003 is too slow (0.875); 0.01 is unstable (training loss lower but validation loss higher). |
| batch size | 32 | 128, 512 | with a fixed number of epochs, smaller batches make more updates: 32 gives 0.924, 128 gives 0.920, 512 gives 0.860, and the cost per epoch is about the same. The random search agrees. |
| architecture | 128 -> 64 | 32, 64, 128, 64 -> 32 | accuracy grows with capacity then saturates (0.892, 0.907, 0.908, 0.920, 0.916). The largest network overfits on 24 000 positions (train 0.988, validation 0.920). 128 -> 64 is used because the final training set is much larger (160 000 positions) and the network is regularised. |
| overfitting | dropout 0.3 + L2 1e-4 + early stopping (patience 5) + large dataset | each alone, dropout 0.1 | on 6 000 positions the plain network reaches train 1.000 / validation 0.843 and L2 improves it (0.863). On the chosen architecture, the combination batch 32 + L2 + dropout 0.3 has the best validation loss (0.172 vs 0.207 for the previous setting). Early stopping restores the best weights. |
| input features | 971 with attack maps | raw 837 (pieces, turn, castling, en passant) | check is a geometric property that a small network learns poorly from raw squares, so attack and king features are given as inputs. This is feature engineering, the network still learns the labelling. The raw-only encoding was not benchmarked. |

### Random search (bonus)

`bonus/hyperparameter_search.py` samples learning rate, architecture, batch
size, activation, dropout and L2 together (10 trials, 8 epochs, 24 000
positions). Results in `logs/hyperparameter_search.json`:

| rank | val loss | val accuracy | lr | hidden | batch | activation | dropout | L2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.167 | 0.938 | 0.0014 | 32 | 32 | relu | 0.3 | 1e-4 |
| 2 | 0.177 | 0.934 | 0.0048 | 32 | 64 | leaky ReLU | 0.1 | 1e-4 |
| 3 | 0.184 | 0.929 | 0.0022 | 32 | 32 | leaky ReLU | 0.1 | 1e-3 |
| 5 | 0.212 | 0.918 | 0.0019 | 128 -> 64 | 256 | tanh | 0.3 | 0 |
| 10 | 0.351 | 0.859 | 0.0007 | 64 | 256 | relu | 0.1 | 1e-3 |

Reading: the best runs all use a small batch (32 or 64), and the worst a large
batch or a low learning rate. With a fixed budget of 8 epochs, the batch size
dominates because it sets the number of updates; the small `[32]` network is
not better in itself, it is fast to train. This agrees with the one-at-a-time
sweeps (batch 32 best, 512 worst, lr 0.0003 too slow).

## What we tried and what did not work

- Constant weight scale 0.01: XOR stuck at loss 0.25 (vanishing signal). Fixed by Xavier / He.
- Sigmoid in hidden layers: clearly worse than ReLU / tanh.
- Plain SGD: slow, needs a large learning rate and many epochs.
- Large batches (512) and low learning rates (0.0003): not enough updates in the epoch budget.
- Dropout 0.3 alone on a small dataset: no real gain; it helped only combined with L2 and a small batch on the larger network.
- The gain of the final training over the previous one is small on held-out data (95.8 % to 96.3 %), much less than the 24 000-position comparison suggested.

## Limits and honest caveats

- Every benchmark variant was run once (seed 0). Differences below about 0.5
  point are noise. Multi-seed runs were not done.
- The sweeps change one variable at a time, so interactions are not measured;
  the random search covers some of them but with only 10 trials.
- The final configuration comes from a comparison of six combinations on 24 000
  positions with one seed, whose curves oscillate: part of the advantage of the
  chosen one may be noise.
- The validation loss of the final training jumps at epoch 11 before recovering
  (early stopping returned the weights of epoch 7).
- The test set comes from the same generator as the training set. The training
  data is balanced; real games contain mostly `Nothing`, so precision on rare
  classes would be lower there.
- Stalemate is not handled (no stalemate position in `assets/`).

## Ideas not done

- Features that give the network what it lacks (blocking and capturing the checking piece), rather than more training, to reduce the check / checkmate confusion; more positions (we used 200 000 of about 720 000).
- A learning-rate schedule.
- Features that simulate blocking and capturing the checking piece.
- Batch normalisation, class weights, learning-rate schedule.
- Multi-seed benchmarks and a larger random search.
- Stalemate (needs positions that are not in `assets/`).
- Parallelism (multi-processing for the benchmarks is already used; the training itself is single-process).

## If we started again

We would generate our own labelled positions with a chess engine (to include
stalemate and realistic class proportions), and keep a held-out test set from a
different source. We would also benchmark the raw encoding against the
engineered features, and set the batch size from the start with the random
search. A convolutional layer over the board would be a natural alternative,
at the cost of a more complex backward pass.

## Data and privacy (GDPR)

The data is made of chess positions (FEN strings) and their labels. It contains
no personal data. The project does not collect or send anything.

## Run the tests

From the repository root:

```sh
python -m unittest discover -s tests -v
```

See [docs/test.md](docs/test.md) for the test commands and coverage,
[docs/analyser.md](docs/analyser.md) for the analyzer workflow, and
[docs/nn.md](docs/nn.md) for the neural-network components.
