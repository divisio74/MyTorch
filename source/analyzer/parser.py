"""
    parser.py
    ...
"""

from source.common.constants import (
    PIECES,
    LABELS,
    INT_STRING
)


class Entry:
    def __init__(self : 'Entry', fen: str, placement: str, turn: str, castling: str, en_passant: str, label : str = None) -> None:
        self.fen = fen
        self.placement = placement
        self.turn = turn
        self.castling = castling
        self.en_passant = en_passant
        self.label = label


def check_placement(placement: str) -> None:
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"invalid FEN: expected 8 ranks, got {len(ranks)}")
    for rank in ranks:
        total = 0
        previous_was_digit = False
        for char in rank:
            if char in INT_STRING:
                if previous_was_digit:
                    raise ValueError(f"invalid FEN: adjacent empty-square counts in rank '{rank}'")
                total += int(char)
                previous_was_digit = True
            elif char in PIECES:
                total += 1
                previous_was_digit = False
            else:
                raise ValueError(f"invalid FEN: bad character '{char}'")
        if total != 8:
            raise ValueError(f"invalid FEN: rank '{rank}' does not have 8 squares")
    if placement.count("K") != 1 or placement.count("k") != 1:
        raise ValueError("invalid FEN: each side needs exactly one king")


def parse_line(line: str) -> Entry:
    tok = line.split()

    if len(tok) < 6:
        raise ValueError(f"invalid FEN line: '{line.strip()}'")
    placement, turn, castling, en_passant, halfmove, fullmove = tok[:6]
    rest = tok[6:]

    check_placement(placement)
    if turn not in ("w", "b"):
        raise ValueError(f"invalid FEN: bad side to move '{turn}'")
    if castling != "-" and (
        not set(castling) <= set("KQkq")
        or len(set(castling)) != len(castling)
        or [ "KQkq".index(char) for char in castling ]
        != sorted("KQkq".index(char) for char in castling)
    ):
        raise ValueError(f"invalid FEN: bad castling field '{castling}'")
    if en_passant != "-" and not (len(en_passant) == 2 and en_passant[0] in "abcdefgh" and en_passant[1] in "36"):
        raise ValueError(f"invalid FEN: bad en passant field '{en_passant}'")
    if not halfmove.isdecimal() or not fullmove.isdecimal() or int(fullmove) < 1:
        raise ValueError("invalid FEN: move counters must be non-negative integers and fullmove must be positive")

    label = None
    if rest:
        key = " ".join(rest).strip().lower()
        if key not in LABELS:
            raise ValueError(f"invalid label '{' '.join(rest)}'")
        label = LABELS[key]

    fen = f"{placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    return Entry(fen, placement, turn, castling, en_passant, label)


def parse_file(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = [(number, line) for number, line in enumerate(file, 1) if line.strip()]
    except OSError as err:
        raise ValueError(f"cannot read '{path}': {err.strerror}")
    if not lines:
        raise ValueError(f"'{path}' is empty")
    entries = []
    for number, line in lines:
        try:
            entries.append(parse_line(line))
        except ValueError as err:
            raise ValueError(f"{path}:{number}: {err}")
    return entries
