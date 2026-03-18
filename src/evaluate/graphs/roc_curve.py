from sklearn.metrics import roc_curve, auc
from src.evaluate.graphs.utils import *
import matplotlib.pyplot as plt
from configs.settings import *
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
import os


def main(model_name, test_ds=None, model=None):
    if test_ds is None or model is None:
        test_ds, model = load_test_data(model_name)

    all_true_cls = []
    all_pred_cls = []

    for images, labels in tqdm(test_ds, desc="Processing batches", ascii=True):
        preds_batch, _ = model.predict(images, verbose=0)
        all_true_cls.append(labels[0])
        all_pred_cls.append(tf.convert_to_tensor(preds_batch, dtype=tf.float32))

    y_true = tf.concat(all_true_cls, axis=0).numpy()
    y_score = tf.concat(all_pred_cls, axis=0).numpy()

    n_classes = y_true.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], color='b', lw=2, label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], color='r', lw=2, linestyle='--', label=f"Macro-average ROC (AUC = {roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"Micro & Macro-average ROC Curves of {model_name}", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)

    save_dir = f"{SAVE_GRAPH_PATH}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/roc_curve_{model_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate.graphs.roc_curve <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
