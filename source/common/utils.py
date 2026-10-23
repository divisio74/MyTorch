"""
    utils.py

"""

from source.common.constants import MAIN_ACCEPTED_FLAGS, ERROR, VALID_LABELS
import sys
import os.path

def is_help_flag(arg : str, n_arg : int) -> bool:
    if arg == MAIN_ACCEPTED_FLAGS[2] and n_arg == 2:
        return True
    return False

def data_files_exist(files_path : tuple[str, str]) -> bool:
    loadfile = os.path.isfile(files_path[0])
    chessfile = os.path.isfile(files_path[1])
    if (loadfile and chessfile):
        return True
    return False

def get_data_files(files_path : list[str]) -> tuple[str, str]:
    if len(files_path) == 4:
        return (files_path[2], files_path[3])
    return (files_path[0], files_path[1])

def check_network_file(path : str) -> None:
    if os.path.getsize(path) == 0:
        raise ValueError(f"{path}: the neural network file is empty")

def check_fen(fen : list[str]) -> None:
    if len(fen) != 6:
        raise ValueError("a FEN needs 6 fields")
    ranks = fen[0].split("/")
    if len(ranks) != 8:
        raise ValueError("a FEN needs 8 ranks")
    for rank in ranks:
        if any(c not in "12345678pnbrqkPNBRQK" for c in rank):
            raise ValueError(f"invalid character in rank '{rank}'")
        if sum(int(c) if c.isdigit() else 1 for c in rank) != 8:
            raise ValueError(f"rank '{rank}' is not 8 squares wide")
    if fen[0].count("K") != 1 or fen[0].count("k") != 1:
        raise ValueError("each side needs exactly one king")
    if fen[1] not in ("w", "b"):
        raise ValueError("active color must be 'w' or 'b'")
    if not fen[4].isdigit() or not fen[5].isdigit():
        raise ValueError("move counters must be integers")

def check_chess_file(path : str, need_label : bool) -> None:
    nb_boards = 0
    with open(path, encoding="utf-8") as file:
        for n, line in enumerate(file, 1):
            tokens = line.split()
            if not tokens:
                continue
            try:
                check_fen(tokens[:6])
                label = " ".join(tokens[6:])
                if need_label and not label:
                    raise ValueError("expected output is missing")
                if label and label not in VALID_LABELS:
                    raise ValueError(f"unknown expected output '{label}'")
            except ValueError as err:
                raise ValueError(f"{path}:{n}: {err}") from None
            nb_boards += 1
    if nb_boards == 0:
        raise ValueError(f"{path}: no chessboard found")

def is_train_flag(arg : str, n_arg : int, files_path : tuple[str, str]) -> bool:
    if arg == MAIN_ACCEPTED_FLAGS[1] and (n_arg == 4 or n_arg == 6):
        if n_arg == 6:
            if files_path[0] != "--save":
                print("Expected '--save SAVEFILE' after --train", file=sys.stderr)
                sys.exit(ERROR)
            if not os.path.isdir(os.path.dirname(files_path[1]) or "."):
                print("The directory of SAVEFILE does not exist", file=sys.stderr)
                sys.exit(ERROR)
        data_files = get_data_files(files_path)
        if data_files_exist(data_files):
            check_network_file(data_files[0])
            check_chess_file(data_files[1], need_label=True)
            return True
        else:
            print("One of the given files does not actually exist", file=sys.stderr)
            sys.exit(ERROR)
    return False

def is_predict_flag(arg : str, n_arg : int, files_path : tuple[str, str]) -> bool:
    if arg == MAIN_ACCEPTED_FLAGS[0] and n_arg == 4:
        if data_files_exist(files_path):
            check_network_file(files_path[0])
            check_chess_file(files_path[1], need_label=False)
            return True
        else:
            print("One of the given files does not actually exist", file=sys.stderr)
            sys.exit(ERROR)
    return False