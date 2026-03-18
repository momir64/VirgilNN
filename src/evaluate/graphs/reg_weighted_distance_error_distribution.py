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


def weighted_geo_average(probabilities, grid_centers):
    deg2rad, rad2deg = np.pi / 180.0, 180.0 / np.pi
    lat_rad = grid_centers[:, 0] * deg2rad
    lon_rad = grid_centers[:, 1] * deg2rad

    x = tf.cos(lat_rad) * tf.cos(lon_rad)
    y = tf.cos(lat_rad) * tf.sin(lon_rad)
    z = tf.sin(lat_rad)
    xyz = tf.stack([x, y, z], axis=1)
    weighted_xyz = tf.linalg.matmul(probabilities, xyz)
    norm = tf.norm(weighted_xyz, axis=1, keepdims=True)
    weighted_xyz /= norm

    lat = tf.asin(weighted_xyz[:, 2]) * rad2deg
    lon = tf.atan2(weighted_xyz[:, 1], weighted_xyz[:, 0]) * rad2deg
    return tf.stack([lat, lon], axis=1)


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    grid_centers = np.array(mu.get_grid_cell_centers(), dtype=np.float32)

    all_true_coords = []
    all_pred_coords = []
    all_pred_softmax = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        y_cls_true, y_reg_true = labels

        if isinstance(images, tuple):
            model_inputs = list(images)
        else:
            model_inputs = images

        preds_cls, preds_reg = model.predict(model_inputs, verbose=0)

        all_true_coords.append(y_reg_true)
        all_pred_coords.append(tf.convert_to_tensor(preds_reg, dtype=tf.float32))
        all_pred_softmax.append(preds_cls)

    true_coords = tf.concat(all_true_coords, axis=0)
    pred_coords = tf.concat(all_pred_coords, axis=0)
    pred_softmax = tf.concat(all_pred_softmax, axis=0)
    pred_softmax = reverse_lex_order_softmax_tensor(pred_softmax)

    weighted_latlon = weighted_geo_average(pred_softmax, grid_centers)

    dist_model = ut.haversine_distance(true_coords, pred_coords, reduce_mean=False)
    dist_weighted = ut.haversine_distance(true_coords, weighted_latlon, reduce_mean=False)
    diff = (dist_model - dist_weighted).numpy()

    q_low, q_high = np.quantile(diff, [0.01, 0.99])
    diff_clipped = diff[(diff >= q_low) & (diff <= q_high)]

    plt.figure(figsize=(10, 6))
    plt.hist(diff_clipped, bins=72, color="lightgreen", edgecolor="black")
    plt.xlabel("Difference in Distance Errors (km)")
    plt.ylabel("Number of Samples")
    plt.title(f"Distance Error Difference: Regression Model - Weighted Softmax ({model_name})")
    plt.grid(True, alpha=0.3)

    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/distance_error_difference_weighted_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.reg_weighted_distance_error_distribution <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
