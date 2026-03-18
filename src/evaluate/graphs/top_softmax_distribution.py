from src.evaluate.graphs.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
import os

def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    all_top_probs = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        softmax_probs, _ = model.predict(images, verbose=0)
        top_probs = tf.reduce_max(softmax_probs, axis=1)
        all_top_probs.append(top_probs)

    top_probs_all = tf.concat(all_top_probs, axis=0).numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(top_probs_all, bins=50, color="lightgreen", edgecolor="black")
    plt.xlabel("Top Softmax Probability")
    plt.ylabel("Number of Samples")
    plt.title(f"Distribution of Top Softmax Probabilities ({model_name})")
    plt.grid(True, alpha=0.3)

    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/top_softmax_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()

    print("Mean top probability:", np.mean(top_probs_all))
    print("Min top probability:", np.min(top_probs_all))
    print("Max top probability:", np.max(top_probs_all))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m top_softmax_distribution <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
