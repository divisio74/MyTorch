# Running the tests

Run commands from the repository root (the directory containing
`my_torch_analyzer` and `tests`).

## Prerequisites

Install the project dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run the complete test suite

```sh
python -m unittest discover -s tests -v
```

The suite includes:

- Analyzer CLI behavior, including help, train, predict, and invalid arguments.
- FEN parsing, label validation, and line-numbered file errors.
- Model initialization, prediction, training, data splitting, and model saving.
- CSV and image training-artifact writers.

The neural-network gradient checks are in `tests/test_gradient.py`. Run them
directly with:

```sh
python tests/test_gradient.py
```

The analyzer tests can also be run on their own:

```sh
python -m unittest discover -s tests -p "test_analyzer.py" -v
```

NumPy and Matplotlib must be installed before running these tests. The project
does not require pytest for its current test suite.

## Try the CLI manually

After installing dependencies, display the usage text:

```sh
python my_torch_analyzer --help
```

To try prediction or training, prepare a model path and a UTF-8 text file of
positions. Every non-empty line begins with a six-field FEN. Add a supported
label after the FEN for training. For example, save this as `boards.txt`:

```text
4k3/8/8/8/8/8/8/4K3 w - - 0 1 Nothing
```

Then run:

```sh
python my_torch_analyzer --train model.json boards.txt
python my_torch_analyzer --predict model.json boards.txt
```

The `LOADFILE` must already exist. If its content is not a readable model JSON,
the current loader initializes a default model; training then saves the result
to `LOADFILE`. To keep the input model unchanged and write to a new path, use:

```sh
python my_torch_analyzer --train --save trained.json model.json boards.txt
```

For the complete input format and workflow, see [analyser.md](analyser.md).
