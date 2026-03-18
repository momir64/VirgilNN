from src.evaluate.graphs.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import sys

def compute_entropy(probabilities):
    return -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-9), axis=1)

def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    all_pred_softmax = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        preds_cls, _ = model.predict(images, verbose=0)
        all_pred_softmax.append(preds_cls)

    pred_softmax = tf.concat(all_pred_softmax, axis=0)

    probabilities = tf.cast(pred_softmax, tf.float32)
    probabilities = tf.clip_by_value(probabilities, 1e-12, 1.0)
    entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=1)
    entropy = tf.where(tf.math.is_finite(entropy), entropy, tf.zeros_like(entropy))
    entropy = entropy.numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=50, color="lightgreen", edgecolor="black")
    plt.xlabel("Softmax Entropy (nats)")
    plt.ylabel("Number of Samples")
    plt.title(f"Entropy Distribution of Classifier Softmax Outputs ({model_name})")
    plt.grid(True, alpha=0.3)

    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/softmax_entropy_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m softmax_entropy_distribution <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
