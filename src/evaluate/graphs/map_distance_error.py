from src.evaluate.eval_utils import reverse_lex_order_softmax
from src.evaluate.map_utils import draw_map
from src.evaluate.graphs.utils import *
from collections import defaultdict
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

    cell_errors = defaultdict(list)

    for batch in tqdm(test_ds, desc="Processing batches", ascii=True):
        images, (y_cls, y_reg) = batch
        pred_cls, pred_coords = model.predict(images, verbose=0)
        true_cls = tf.argmax(y_cls, axis=1).numpy()

        true_latlon = tf.cast(y_reg, tf.float32)
        pred_latlon = tf.cast(pred_coords, tf.float32)
        batch_km = ut.haversine_distance(true_latlon, pred_latlon, reduce_mean=False).numpy()
        for cls_idx, km in zip(true_cls, batch_km):
            cell_errors[cls_idx].append(km)

    mean_error = {idx: np.mean(errors) for idx, errors in cell_errors.items()}
    max_error = max(mean_error.values(), default=1.0)
    normalized_error = {idx: val / max_error for idx, val in mean_error.items()}
    normalized_error = reverse_lex_order_softmax(normalized_error)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("lightskyblue")
    draw_map(ax, lat=None, lon=None, show_grid=True, softmax=normalized_error)
    plt.title(f"Distance Error per Cell of {model_name}", fontsize=14)
    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/map_{model_name}_distance.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.map_distance_error <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
