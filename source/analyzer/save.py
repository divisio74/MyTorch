"""
    save.py
    ...
"""
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from source.common.constants import ERROR, NOERROR, TEMP_STRING

matplotlib.use("Agg")

def validated_losses(loss_history : list[float]) -> list[float]:
    if not loss_history:
        raise ValueError("loss_history must contain at least one value")
    if not all(math.isfinite(loss) for loss in loss_history):
        raise ValueError("loss_history values must be finite")
    return loss_history


def unique_naming(prefix : str, extension : str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}{TEMP_STRING}.{extension}"


def save_as_csv(loss_history : list[float], output_dir : str | Path = "outputs", prefix : str = "training_loss") -> Path:
    losses = validated_losses(loss_history)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / unique_naming(prefix, "csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("epoch", "loss"))
        writer.writerows(enumerate(losses, start=1))

    return csv_path


def save_as_img(loss_history : list[float], output_dir : str | Path = "outputs", title : str = "Training loss", y_label : str = "Loss", prefix : str = "training_loss") -> Path:
    losses = validated_losses(loss_history)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graph_path = output_path / unique_naming(prefix, "png")
    epochs = range(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", color="#2563eb", linewidth=2, markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, color="#ddd")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)

    return graph_path
