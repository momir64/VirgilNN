import sys

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model.data_loader_single import load_data_splits, make_dataset
from src.model.middle_integration.architecture_middle_single import build_single_view_model
from configs.settings import *
import os, pickle

from src.model.utils import HistorySaverCallback, enable_mixed_precision_if_supported


def main(model_name):
    data_splits = load_data_splits()
    image_shape = data_splits["image_shape"]
    train_ds = make_dataset(*data_splits["train"], image_shape[:2])
    val_ds = make_dataset(*data_splits["validation"], image_shape[:2])
    num_classes = data_splits["train"][1].shape[1]

    model = build_single_view_model(image_shape, num_classes)
    callbacks = [
        ModelCheckpoint(MODEL_PATH(model_name), save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=5),
        EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
        HistorySaverCallback(model_name)
    ]

    if USE_MIXED_PRECISION:
        enable_mixed_precision_if_supported()

    print('Training...')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_SINGLE,
        callbacks=callbacks
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.model.middle_integration.train_middle_single <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)