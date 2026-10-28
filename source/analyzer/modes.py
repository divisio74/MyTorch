"""
    modes.py
    ...
"""

import os
from source.common.constants import MAIN_ACCEPTED_FLAGS
from source.analyzer.parser import parse_file, parse_line


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
        raise ValueError(f"invalid LOADFILE: '{path}' is empty")


def check_fen(fen : list[str]) -> None:
    parse_line(" ".join(fen))


def check_chess_file(path : str, need_label : bool) -> None:
    nb_boards = 0
    with open(path, encoding="utf-8") as file:
        for n, line in enumerate(file, 1):
            tokens = line.split()
            if not tokens:
                continue
            try:
                entry = parse_line(line)
                if need_label and entry.label is None:
                    raise ValueError("expected output is missing")
            except ValueError as err:
                raise ValueError(f"{path}:{n}: {err}")
            nb_boards += 1
    if nb_boards == 0:
        raise ValueError(f"invalid CHESSFILE: '{path}' is empty")


def is_train_flag(arg : str, n_arg : int, files_path : tuple[str, str]) -> bool:
    if arg == MAIN_ACCEPTED_FLAGS[1] and (n_arg == 4 or n_arg == 6):
        if n_arg == 6:
            if files_path[0] != "--save":
                raise ValueError("Expected '--save SAVEFILE' after --train")
            if not os.path.isdir(os.path.dirname(files_path[1]) or "."):
                raise ValueError(f"invalid SAVEFILE: directory '{os.path.dirname(files_path[1])}' does not exist")
        data_files = get_data_files(files_path)
        if data_files_exist(data_files):
            check_network_file(data_files[0])
            check_chess_file(data_files[1], need_label=True)
            return True
        else:
            raise ValueError("One of the given files does not actually exist")
    return False


def is_predict_flag(arg : str, n_arg : int, files_path : tuple[str, str]) -> bool:
    if arg == MAIN_ACCEPTED_FLAGS[0] and n_arg == 4:
        if data_files_exist(files_path):
            check_network_file(files_path[0])
            check_chess_file(files_path[1], need_label=False)
            return True
        else:
            raise ValueError("One of the given files does not actually exist")
    return False


def split_train_args(files_path : list[str]) -> tuple[str, str, str]:
    args = list(files_path)
    save = None

    if args and args[0] == "--save":
        if len(args) != 4:
            raise ValueError("usage: --train [--save SAVEFILE] LOADFILE CHESSFILE")
        save, args = args[1], args[2:]

    if len(args) != 2:
        raise ValueError("usage: --train [--save SAVEFILE] LOADFILE CHESSFILE")

    return args[0], args[1], save


def run_predict(loadfile : str, chessfile : str) -> None:
    net = 0#load(loadfile)
    entries = parse_file(chessfile)
    print(net)


def run_train(loadfile : str, chessfile : str, savefile : str = None, val_ratio : float = 0.1, seed : int = 0) -> None:
    net = 0#load(loadfile)
    entries = parse_file(chessfile)
    print(net)
