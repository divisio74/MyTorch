# MyTorch chess analyzer

A small NumPy-based neural-network project with a command-line analyzer for chess
positions written in FEN notation.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt` (`numpy` and `matplotlib`)

Install the dependencies from the repository root:

```sh
python -m pip install -r requirements.txt
```

## Run the analyzer

The command-line entry point is `my_torch_analyzer`. Run it with Python:

```sh
python my_torch_analyzer --help
python my_torch_analyzer --predict model.json positions.txt
python my_torch_analyzer --train model.json labeled-positions.txt
python my_torch_analyzer --train --save trained.json model.json labeled-positions.txt
```

Training without `--save` updates the model at `LOADFILE`. With `--save`, the
updated model is written to `SAVEFILE` and `LOADFILE` is left unchanged.

Each non-empty line in a chess input file starts with a six-field FEN position.
Training lines must end with one of the supported labels; prediction labels are
optional. For example:

```text
4k3/8/8/8/8/8/8/4K3 w - - 0 1 Nothing
```

The labels are `Nothing`, `Check`, `Checkmate`, `Stalemate`, `Check White`,
`Check Black`, `Checkmate White`, and `Checkmate Black`.

## Run the tests

From the repository root:

```sh
python -m unittest discover -s tests -v
```

See [docs/test.md](docs/test.md) for the test commands and coverage,
[docs/analyser.md](docs/analyser.md) for the analyzer workflow, and
[docs/nn.md](docs/nn.md) for the neural-network components.

## Current scope and future work

The analyzer currently uses a small hand-written NumPy model and does not yet
report validation metrics or training-loss artifacts as part of its training
workflow. Potential next steps include logging each run and its random seed,
benchmarking and documenting hyperparameter choices, measuring overfitting,
automatically promoting a model only when it beats the validated version,
reviewing data-privacy requirements, and exploring parallelism.
