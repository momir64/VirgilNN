from src.evaluate.graphs.utils import *
import matplotlib.pyplot as plt
from configs.settings import *
import src.model.utils as ut
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    distances = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        true_coords = tf.cast(labels[1], tf.float32)
        pred_coords = tf.cast(model.predict(images, verbose=0)[1], tf.float32)
        batch_distances = ut.haversine_distance(true_coords, pred_coords, reduce_mean=False).numpy()
        distances.extend(batch_distances)

    distances = np.array(distances)

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=72, color="lightgreen", edgecolor="black")
    plt.xlabel("Distance Error (km)")
    plt.ylabel("Number of Guesses")
    plt.title(f"Distance Distribution of {model_name}")
    plt.grid(True, alpha=0.3)
    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/distance_distribution_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.distance_distribution <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
