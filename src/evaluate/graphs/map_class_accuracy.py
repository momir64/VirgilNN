from src.evaluate.eval_utils import reverse_lex_order_softmax
from src.evaluate.map_utils import draw_map
from src.evaluate.graphs.utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
from configs.settings import *
import tensorflow as tf
from tqdm import tqdm
import sys


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in tqdm(test_ds, desc="Processing batches", ascii=True):
        x, (y_cls, y_reg) = batch
        pred_cls, _ = model.predict(x, verbose=0)
        true_cls = tf.argmax(y_cls, axis=1).numpy()
        pred_cls = tf.argmax(pred_cls, axis=1).numpy()
        for t, p in zip(true_cls, pred_cls):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1

    grid_acc = {i: class_correct[i] / class_total[i] for i in class_total}
    max_acc = max(grid_acc.values(), default=1.0)
    normalized = {i: acc / max_acc for i, acc in grid_acc.items()}
    normalized = reverse_lex_order_softmax(normalized)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("lightskyblue")
    draw_map(ax, lat=None, lon=None, show_grid=True, softmax=normalized)
    plt.title(f"Classification Accuracy per Cell of {model_name}", fontsize=14)
    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/map_{model_name}_classification.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.map_class_accuracy <model_name>")

    model_name = sys.argv[1]
    main(model_name)
