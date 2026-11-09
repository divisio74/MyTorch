import numpy as np

PIECES = "PNBRQKpnbrqk"
CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
N_FEATURES = 971

KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
STRAIGHTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def parse_board(placement):
    board = []
    for rank in placement.split("/"):
        for char in rank:
            if char.isdigit():
                board.extend([-1] * int(char))
            else:
                board.append(PIECES.index(char))
    return board


def is_white(piece):
    return piece < 6


def attack_counts(board, white):
    counts = [0] * 64
    for square, piece in enumerate(board):
        if piece < 0 or is_white(piece) != white:
            continue
        row, col = divmod(square, 8)
        kind = piece % 6
        if kind == 0:
            step = -1 if white else 1
            for dc in (-1, 1):
                r, c = row + step, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    counts[r * 8 + c] += 1
        elif kind == 1 or kind == 5:
            for dr, dc in (KNIGHT if kind == 1 else KING):
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    counts[r * 8 + c] += 1
        else:
            directions = []
            if kind in (2, 4):
                directions += DIAGONALS
            if kind in (3, 4):
                directions += STRAIGHTS
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    counts[r * 8 + c] += 1
                    if board[r * 8 + c] >= 0:
                        break
                    r += dr
                    c += dc
    return counts


def king_info(board, white):
    king = board.index(5 if white else 11)
    enemy = attack_counts(board, not white)
    checkers = enemy[king]
    without_king = list(board)
    without_king[king] = -1
    enemy = attack_counts(without_king, not white)
    row, col = divmod(king, 8)
    escapes = 0
    for dr, dc in KING:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            piece = board[r * 8 + c]
            if (piece < 0 or is_white(piece) != white) and enemy[r * 8 + c] == 0:
                escapes += 1
    return checkers, escapes


def encode(fen):
    fields = fen.split()
    board = parse_board(fields[0])
    vector = np.zeros(N_FEATURES, dtype=np.float32)
    for square, piece in enumerate(board):
        if piece >= 0:
            vector[piece * 64 + square] = 1
    vector[768] = fields[1] == "w"
    for i, flag in enumerate("KQkq"):
        vector[769 + i] = flag in fields[2]
    if fields[3] != "-":
        vector[773 + (8 - int(fields[3][1])) * 8 + "abcdefgh".index(fields[3][0])] = 1
    white_attacks = attack_counts(board, True)
    black_attacks = attack_counts(board, False)
    for square in range(64):
        vector[837 + square] = white_attacks[square] > 0
        vector[901 + square] = black_attacks[square] > 0
    white_checkers, white_escapes = king_info(board, True)
    black_checkers, black_escapes = king_info(board, False)
    vector[965] = white_checkers > 0
    vector[966] = black_checkers > 0
    vector[967] = white_checkers / 2
    vector[968] = black_checkers / 2
    vector[969] = white_escapes / 8
    vector[970] = black_escapes / 8
    return vector


def encode_many(fens):
    return np.array([encode(fen) for fen in fens], dtype=np.float32)


def label_index(label):
    return CLASSES.index(label)
