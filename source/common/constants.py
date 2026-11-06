"""
    constants.py
    ...
"""

MAIN_ACCEPTED_FLAGS = ["--predict", "--train", "--help"]

ERROR = 84
NOERROR = 0

LOOP = 1

HELP_TEXT = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
        
        --train     Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.

        --predict   Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.
        
        --save      Save neural network into SAVEFILE. Only works in train mode.
        
        LOADFILE    File containing an artificial neural network
        CHESSFILE   File containing chessboards
"""

WDC = 0.01

LABEL_ORDER = (
    "Nothing",
    "Check",
    "Checkmate",
    "Stalemate",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
)
LABELS = {label.lower(): label for label in LABEL_ORDER}
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}
INT_STRING = "12345678"

BOARD_SIDE = 8

BOARD_SQUARES = 64 # BOARD_SIDE * BOARD_SIDE

MODEL_FORMAT = "mytorch-chess-v1"

FEN_PIECE_VALUES = {
    "P" : 1.0,
    "N" : 2.0,
    "B" : 3.0,
    "R" : 4.0,
    "Q" : 5.0,
    "K" : 6.0,
    "p" : -1.0,
    "n" : -2.0,
    "b" : -3.0,
    "r" : -4.0,
    "q" : -5.0,
    "k" : -6.0,
}
PIECES = "".join(FEN_PIECE_VALUES)

TRAINING_EPOCHS = 100
LEARNING_RATE = 0.01

TEMP_STRING = ".temp"