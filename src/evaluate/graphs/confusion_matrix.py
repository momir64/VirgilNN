from sklearn.metrics import confusion_matrix
from src.evaluate.graphs.utils import *
import matplotlib.pyplot as plt
from configs.settings import *
import seaborn as sns
from tqdm import tqdm
import numpy as np
import sys


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    y_true = []
    y_pred = []

    for batch in tqdm(test_ds, desc="Processing batches", ascii=True):
        images, labels = batch
        cls_labels = labels[0]
        preds, _ = model.predict(images, verbose=0)
        y_true.extend(np.argmax(cls_labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm, cmap="magma", annot=cm, fmt="d", cbar=True, square=True,
                xticklabels=False, yticklabels=False, annot_kws={"fontsize": 4})
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.title(f"Confusion Matrix of {model_name}")
    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/confusion_matrix_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.confusion_matrix <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
