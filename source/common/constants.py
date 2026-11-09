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

VALID_LABELS = {"Nothing", "Check", "Checkmate", "Stalemate",
                "Check White", "Check Black",
                "Checkmate White", "Checkmate Black"}

LABELS = {"nothing": "Nothing", "check": "Check",
          "checkmate": "Checkmate",
          "stalemate": "Stalemate",
          "check white": "Check White", "check black": "Check Black",
          "checkmate white": "Checkmate White",
          "checkmate black": "Checkmate Black"}

PIECES = "pnbrqkPNBRQK"
INT_STRING = "12345678"

TRAINING_EPOCHS = 100
LEARNING_RATE = 0.01