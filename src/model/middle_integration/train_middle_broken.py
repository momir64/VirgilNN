from src.model.utils import enable_mixed_precision_if_supported, HistorySaverCallback, AccuracyMeasurerCallback
from src.model.middle_integration.data_loader_middle import load_data_splits_grouped, make_grouped_dataset
from src.model.middle_integration.architecture_middle import initialize_unfrozen_model
from keras.src.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback
from configs.settings import *
import tensorflow as tf
import numpy as np
import sys
import os


class DelayedModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=True, start_epoch=0, mode='auto'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.start_epoch = start_epoch
        self.best = None
        self.monitor_op = self._get_monitor_op(mode, monitor)
        self.epoch_count = 0

    def _get_monitor_op(self, mode, monitor):
        if mode not in ['auto', 'min', 'max']:
            print(f"Unknown mode '{mode}', defaulting to 'auto'")
            mode = 'auto'

        if mode == 'min' or (mode == 'auto' and 'loss' in monitor):
            self.best = np.inf
            return np.less
        else:
            self.best = -np.inf
            return np.greater

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_count += 1

        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found in logs.")
            return

        if self.epoch_count <= self.start_epoch:
            if self.verbose:
                print(f"Skipping checkpoint at epoch {epoch + 1} (start_epoch={self.start_epoch})")
            return

        if not self.save_best_only:
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: saving model to {self.filepath}")
            self.model.save(self.filepath)
        else:
            if self.monitor_op(current, self.best):
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {self.filepath}")
                self.best = current
                self.model.save(self.filepath)
            else:
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")


class DelayedReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.epochs_seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_seen += 1
        if self.epochs_seen > self.start_epoch:
            super().on_epoch_end(epoch, logs)
        else:
            if self.verbose > 0:
                print(f"Skipping LR reduction check at epoch {epoch + 1} (before start_epoch={self.start_epoch})")


class PeriodicModelCheckpoint(Callback):
    def __init__(self, filepath, period=5, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.period = period
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            file_path = self.filepath.format(epoch=epoch + 1)
            self.model.save(file_path)
            if self.verbose:
                print(f"Epoch {epoch + 1}: saved model to {file_path}")


def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main(frozen_model_name, model_name):
    # reduce_on_plateau = DelayedReduceLROnPlateau(monitor='val_loss', patience=6, verbose=1, start_epoch=10)
    # reduce_on_plateau = ReduceLROnPlateau(patience=8, verbose=1)
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_delta=1e-5,
        cooldown=2,
        verbose=1
    )
    accuracy_measurer_unfrozen = AccuracyMeasurerCallback(model_name + "_unfrozen")
    history_saver_unfrozen = HistorySaverCallback(model_name + "_unfrozen")
    model_save_path = MODEL_PATH(model_name + "_unfrozen")
    ensure_dir_exists(model_save_path)
    # model_saver_unfrozen = DelayedModelCheckpoint(model_save_path, verbose=1, start_epoch=10)
    # model_saver_unfrozen = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)
    model_saver_unfrozen = PeriodicModelCheckpoint(
        filepath=model_save_path.split(".keras")[0] + "_epoch_{epoch:03d}.keras",
        period=2,
        verbose=1
    )

    print("TensorFlow version:", tf.__version__)
    print("Device name:", tf.test.gpu_device_name())
    print("Built with GPU support?", tf.test.is_built_with_gpu_support())
    print("Built with CUDA?", tf.test.is_built_with_cuda())

    if USE_MIXED_PRECISION:
        enable_mixed_precision_if_supported()

    data_splits = load_data_splits_grouped()
    image_shape = data_splits["image_shape"]
    views = data_splits["views_per_sample"]

    train_ds = make_grouped_dataset(*data_splits["train"], image_shape[:2])
    val_ds = make_grouped_dataset(*data_splits["validation"], image_shape[:2])
    test_ds = make_grouped_dataset(*data_splits["test"], image_shape[:2])

    frozen_model_path = MODEL_PATH(frozen_model_name)
    if not os.path.exists(frozen_model_path):
        print(f"Frozen model '{frozen_model_name}' not found at {frozen_model_path}")
        sys.exit(1)

    print(f"Loading frozen model '{frozen_model_name}'")
    model = tf.keras.models.load_model(frozen_model_path, compile=False)
    model = initialize_unfrozen_model(model)

    print("\n--- Fine-tuning unfrozen backbone ---")
    model.fit(
        train_ds,
        initial_epoch=8,
        validation_data=val_ds,
        epochs=EPOCHS_MIDDLE,
        callbacks=[reduce_on_plateau, accuracy_measurer_unfrozen,
                   history_saver_unfrozen, model_saver_unfrozen]
    )

    final_save_path = MODEL_PATH(model_name + "_final")
    ensure_dir_exists(final_save_path)
    model.save(final_save_path)

    print('Evaluation')
    model.evaluate(test_ds)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.model.middle_integration.train_middle_broken <frozen_model_name> <model_name>")
        sys.exit(1)

    frozen_model_name = sys.argv[1]
    model_name = sys.argv[2]
    main(frozen_model_name, model_name)
