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
4. `predict_label` multiplies the vector by the first weight matrix, applies
   `tanh`, multiplies by the second matrix, and selects the label with the
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
   position using a `tanh` hidden layer, a mean-squared-error gradient, and a
   learning rate of `0.01`.
5. Save the updated weights and print the number of training and validation
   entries.

The current validation subset is counted but **not evaluated**: no validation
loss or accuracy is calculated, and it does not affect training or model
selection. Training also does not currently shuffle examples between epochs,
record per-epoch loss, or automatically compare/publish a candidate model.
Those are possible extensions, not current behavior.

## Model files

Models are JSON documents with the format identifier `mytorch-chess-v1`,
dimension metadata, and two weight matrices. The first matrix maps 64 input
features to the hidden layer, and the second maps the hidden layer to the
eight label scores. New default models use reproducible nonzero weights.

The model loader initializes a default model when an existing file is empty,
malformed JSON, or not a JSON object. A JSON object with weights is returned
as-is and checked later by the consumer; for a model you want to retain,
provide a valid model JSON rather than relying on this fallback.

## Implementation boundaries

The analyzer has its own training and prediction implementation in
`source/analyzer/train.py` and `source/analyzer/predict.py`. It does not use the
separate general-purpose classes in `source/nn`. See [nn.md](nn.md) for those
components and how to extend them.
