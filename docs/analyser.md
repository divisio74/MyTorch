# Chess analyzer

The analyzer is a command-line program in `my_torch_analyzer`. Its workflows
are implemented in `source/analyzer`; shared model and FEN-vector utilities
are in `source/common/utils.py`.

## Commands

Run from the repository root:

```sh
python my_torch_analyzer --help
python my_torch_analyzer --predict LOADFILE CHESSFILE
python my_torch_analyzer --train LOADFILE CHESSFILE
python my_torch_analyzer --train --save SAVEFILE LOADFILE CHESSFILE
```

`LOADFILE` must exist. Training saves back to `LOADFILE` unless `--save
SAVEFILE` is supplied. The save directory must already exist.

## Input format and validation

Each non-empty line in `CHESSFILE` contains a FEN position followed optionally
by a label. A FEN has six fields: piece placement, active color, castling
rights, en-passant target, halfmove clock, and fullmove number.

Training requires a label after each FEN. Prediction accepts lines with or
without labels. Labels are matched case-insensitively and must be one of:

```text
Nothing
Check
Checkmate
Stalemate
Check White
Check Black
Checkmate White
Checkmate Black
```

Example:

```text
4k3/8/8/8/8/8/8/4K3 w - - 0 1 Nothing
```

The parser checks FEN field formats, board rank widths, and that there is
exactly one king of each color. It does not check all chess-position legality
or determine whether the supplied label is factually correct.

## Prediction flow

1. `source/analyzer/modes.py` validates the CLI arguments and input files.
2. `source/analyzer/predict.py` loads the model and parses the position file.
3. `fen_to_board_vector` converts the piece-placement field to 64 numeric
   values. White pieces have positive values, black pieces negative values;
   each piece type has a different magnitude.
4. `predict_label` builds the model's `Network` and uses `Network.predict` to
   apply its dense, `tanh`, and dense layers, then selects the label with the
   highest score.
5. The program prints each normalized FEN and its predicted label.

The vector contains only piece placement. It does not encode side to move,
castling rights, en-passant state, or move counters; positions that differ
only in those fields therefore have identical model inputs.

## Training flow

1. Parse all entries and reject any that are missing a label.
2. Split the entries into training and validation subsets (default validation
   ratio `0.1`; the current CLI uses the default seed `0`).
3. Load the model and check that its output dimension matches the eight labels.
4. For each of the 100 configured epochs, train sequentially on each training
   position using `Network.fit`, a `tanh` hidden layer, mean squared error, and
   SGD with a learning rate of `0.01`.
5. Evaluate loss and accuracy on the training and validation subsets, and save
   the updated model plus per-epoch CSV and PNG metric files in `outputs/`.

The validation subset is evaluated but does not affect training or model
selection. Training does not currently shuffle examples between epochs or
automatically compare and publish a candidate model.

## Model files

Models are JSON documents with the format identifier `mytorch-chess-v1`,
dimension metadata, and two weight matrices. The first matrix maps 64 input
features to the hidden layer, and the second maps the hidden layer to the
eight label scores. Trained models also include the two dense-layer bias
vectors; models without biases use zero biases. New default models use
reproducible nonzero weights.

The model loader initializes a default model when an existing file is empty,
malformed JSON, or not a JSON object. A JSON object with weights is returned
as-is and checked later by the consumer; for a model you want to retain,
provide a valid model JSON rather than relying on this fallback.

## Implementation boundaries

The analyzer's training and prediction workflows are in
`source/analyzer/train.py` and `source/analyzer/predict.py`; their model
computation uses the reusable classes in `source/nn`. See [nn.md](nn.md) for
those components and how to extend them.
