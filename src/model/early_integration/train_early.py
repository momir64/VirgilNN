from src.model.utils import AccuracyMeasurerCallback, HistorySaverCallback, enable_mixed_precision_if_supported
from src.model.early_integration.architecture_early_second import initialize_model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.model.data_loader_single import *
from configs.settings import *
import tensorflow as tf
import sys


def main(model_name):
    early_stop = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
    reduce_on_plateau = ReduceLROnPlateau(patience=8)
    accuracy_measurer = AccuracyMeasurerCallback(model_name)
    history_saver = HistorySaverCallback(model_name)
    model_saver = ModelCheckpoint(MODEL_PATH(model_name), verbose=1, save_best_only=True)

    print("TensorFlow version: {}".format(tf.__version__))
    print("Device name: {}".format(tf.test.gpu_device_name()))
    print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
    print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

    if USE_MIXED_PRECISION:
        enable_mixed_precision_if_supported()

    data_splits = load_data_splits()
    image_shape = data_splits["image_shape"]

    train_ds = make_dataset(*data_splits["train"], image_shape[:2])
    val_ds = make_dataset(*data_splits["validation"], image_shape[:2])
    test_ds = make_dataset(*data_splits["test"], image_shape[:2])

    num_classes = data_splits["train"][1].shape[1]
    model = initialize_model(image_shape, num_classes)

    print('Training...')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_EARLY,
        callbacks=[early_stop, reduce_on_plateau, accuracy_measurer, history_saver, model_saver]
    )

    print('Evaluation')
    model.evaluate(test_ds)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.model.early_integration.train_early <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
