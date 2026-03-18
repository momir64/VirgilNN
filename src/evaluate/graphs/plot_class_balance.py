from src.model.middle_integration.data_loader_middle import load_data_splits_grouped
from src.model.data_loader_single import load_data_splits
import matplotlib.pyplot as plt
from configs.settings import *
import numpy as np
import os


def plot_class_balance(name, classes, ax):
    class_counts = np.sum(classes, axis=0)
    n_classes = classes.shape[1]
    ax.bar(range(n_classes), class_counts, color='lightgreen')
    ax.set_xlabel("Class Index", fontsize=6)
    ax.set_ylabel("Number of Samples", fontsize=6)
    ax.set_title(f"{name} Set", fontsize=8)
    ax.set_xticks(range(0, n_classes, max(1, n_classes // 20)))
    ax.set_xticklabels(range(0, n_classes, max(1, n_classes // 20)), fontsize=4, rotation=90)
    for i, count in enumerate(class_counts):
        ax.text(i, count + 0.5, str(int(count)), ha='center', fontsize=2, rotation=90)


def main(grouped=False):
    if grouped:
        data_splits = load_data_splits_grouped()
    else:
        data_splits = load_data_splits()

    sets = {
        "Train": data_splits["train"][1],
        "Validation": data_splits["validation"][1],
        "Test": data_splits["test"][1]
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    for ax, (name, classes) in zip(axes, sets.items()):
        plot_class_balance(name, classes, ax)

    plt.suptitle("Class Balance Across Dataset Splits", fontsize=10)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_dir = f"{SAVE_GRAPH_PATH}"
    os.makedirs(save_dir, exist_ok=True)
    type = "grouped" if grouped else "panorama"
    plt.savefig(f"{save_dir}/class_balance_{type}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
