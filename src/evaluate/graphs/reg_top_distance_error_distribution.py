from src.evaluate.eval_utils import reverse_lex_order_softmax_tensor
from src.evaluate.graphs.utils import *
import src.evaluate.map_utils as mu
import matplotlib.pyplot as plt
from configs.settings import *
import src.model.utils as ut
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys, os


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    grid_centers = np.array(mu.get_grid_cell_centers(), dtype=np.float32)

    all_true_coords = []
    all_pred_coords = []
    all_pred_softmax = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        if isinstance(images, tuple):
            model_inputs = list(images)
        else:
            model_inputs = images

        y_cls_true, y_reg_true = labels
        preds_cls, preds_reg = model.predict(model_inputs, verbose=0)
        all_true_coords.append(y_reg_true)
        all_pred_coords.append(tf.convert_to_tensor(preds_reg, dtype=tf.float32))
        all_pred_softmax.append(preds_cls)

    true_coords = tf.concat(all_true_coords, axis=0)
    pred_coords = tf.concat(all_pred_coords, axis=0)
    pred_softmax = tf.concat(all_pred_softmax, axis=0)
    pred_softmax = reverse_lex_order_softmax_tensor(pred_softmax)

    top_class_indices = tf.argmax(pred_softmax, axis=1)
    top_class_grid = tf.gather(grid_centers, top_class_indices)

    dist_model = ut.haversine_distance(true_coords, pred_coords, reduce_mean=False)
    dist_topclass = ut.haversine_distance(true_coords, top_class_grid, reduce_mean=False)
    diff = (dist_model - dist_topclass).numpy()

    q_low, q_high = np.quantile(diff, [0.01, 0.99])
    diff_clipped = diff[(diff >= q_low) & (diff <= q_high)]

    plt.figure(figsize=(10, 6))
    plt.hist(diff_clipped, bins=72, color="lightgreen", edgecolor="black")
    plt.xlabel("Difference in Distance Errors (km)")
    plt.ylabel("Number of Samples")
    plt.title(f"Distance Error Difference: Regression Model - Top Class Center ({model_name})")
    plt.grid(True, alpha=0.3)

    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/distance_error_difference_top_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.reg_top_distance_error_distribution <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
