from src.model.middle_integration.architecture_middle import initialize_middle_model, initialize_unfrozen_model
from src.model.middle_integration.data_loader_middle import load_data_splits_grouped, make_grouped_dataset
from src.model.utils import enable_mixed_precision_if_supported, HistorySaverCallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model.utils import AccuracyMeasurerCallback
from keras.src.callbacks import ModelCheckpoint
from configs.settings import *
import tensorflow as tf
import sys

def main(step1_model_name, model_name):
    early_stop = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
    reduce_on_plateau = ReduceLROnPlateau(patience=8)
    accuracy_measurer_frozen = AccuracyMeasurerCallback(model_name + "_frozen")
    accuracy_measurer_unfrozen = AccuracyMeasurerCallback(model_name + "_unfrozen")
    history_saver_frozen = HistorySaverCallback(model_name + "_frozen")
    history_saver_unfrozen = HistorySaverCallback(model_name + "_unfrozen")
    model_saver_frozen = ModelCheckpoint(MODEL_PATH(model_name + "_frozen"), verbose=1, save_best_only=True)
    model_saver_unfrozen = ModelCheckpoint(MODEL_PATH(model_name + "_unfrozen"), verbose=1, save_best_only=True)

    print("TensorFlow version: {}".format(tf.__version__))
    print("Device name: {}".format(tf.test.gpu_device_name()))
    print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
    print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

    if USE_MIXED_PRECISION:
        enable_mixed_precision_if_supported()

    data_splits = load_data_splits_grouped()
    image_shape = data_splits["image_shape"]
    views = data_splits["views_per_sample"]

    train_ds = make_grouped_dataset(*data_splits["train"], image_shape[:2])
    val_ds = make_grouped_dataset(*data_splits["validation"], image_shape[:2])
    test_ds = make_grouped_dataset(*data_splits["test"], image_shape[:2])

    num_classes = data_splits["train"][1].shape[1]
    model = initialize_middle_model(
        input_shape=image_shape,
        num_views=views,
        num_classes=num_classes,
        pretrained_backbone_path=MODEL_PATH(step1_model_name),
        backbone_trainable=False
    )

    print('Training...')
    print("\n--- Phase 1: Training with frozen backbone ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=UNFREEZE_EPOCH,
        callbacks=[early_stop, reduce_on_plateau, accuracy_measurer_frozen,
                   history_saver_frozen, model_saver_frozen]
    )

    print("\n--- Unfreezing backbone and recompiling ---")
    model = initialize_unfrozen_model(model)

    print("\n--- Phase 2: Fine-tuning unfrozen backbone ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=UNFREEZE_EPOCH,
        epochs=EPOCHS_MIDDLE,
        callbacks=[early_stop, reduce_on_plateau,
                   accuracy_measurer_unfrozen, history_saver_unfrozen,
                   model_saver_unfrozen]
    )

    # final save for safety reasons
    model.save(MODEL_PATH(model_name + "_final"))

    print('Evaluation')
    model.evaluate(test_ds)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.model.middle_integration.train_middle <step1_model_name> <model_name>")
        sys.exit(1)

    step1_model_name = sys.argv[1]
    model_name = sys.argv[2]
    main(step1_model_name, model_name)
