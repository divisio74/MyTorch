"""
    utils.py
    ...
"""


import csv
from datetime import date
import sys
import math
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt
from source.common.constants import ERROR


matplotlib.use("Agg")


def validated_losses(loss_history : list[float]) -> list[float]:
    losses = [float(loss) for loss in loss_history]

    if not losses:
        raise ValueError("loss_history must contain at least one value")
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("loss_history values must be finite")
    return losses


def save_as_csv(loss_history : list[float], output_dir : str | Path = "outputs") -> Path:
    losses = validated_losses(loss_history)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "training_loss" + date.today().isoformat() + ".temp.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("epoch", "loss"))
        writer.writerows(enumerate(losses, start=1))

    return csv_path


def save_as_img(loss_history : list[float], output_dir : str | Path = "outputs") -> Path:
    losses = validated_losses(loss_history)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graph_path = output_path / "training_loss" + date.today().isoformat() + ".temp.png"

    epochs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, losses, marker="o", color="#2563eb", linewidth=2, markersize=4)
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, color="#ddd")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)

    return graph_path


def save_training_artifacts(loss_history : list[float], output_dir : str | Path = "outputs") -> tuple[Path, Path]:
    try:
        graph_path = save_as_img(loss_history, output_dir)
        csv_path = save_as_csv(loss_history, output_dir)
    except Exception as err:
        print(f"Failed to save training artifacts to {output_dir} due to: {err}")
        sys.exit(ERROR)
    return graph_path, csv_path