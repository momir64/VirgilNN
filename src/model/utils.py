from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from configs.settings import *
import tensorflow as tf
import numpy as np
import pickle
import os


class AccuracyMeasurerCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.metrics = {
            'train_class': [],
            'val_class': [],
            'train_reg': [],
            'val_reg': []
        }
        self.logs_printed = False
        os.makedirs(os.path.dirname(SAVE_MODEL_REGRESSION_GRAPH_PATH(model_name)), exist_ok=True)
        os.makedirs(os.path.dirname(SAVE_MODEL_ACCURACY_GRAPH_PATH(model_name)), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.logs_printed:
            print("Log keys:", list(logs.keys()))
            self.logs_printed = True

        self.metrics['train_class'].append(logs.get('classification_accuracy', 0.0))
        self.metrics['val_class'].append(logs.get('val_classification_accuracy', 0.0))

        reg_err = logs.get('coordinates_haversine') or logs.get('coordinates_accuracy', 0.0)
        val_reg_err = logs.get('val_coordinates_haversine') or logs.get('val_coordinates_accuracy', 0.0)

        self.metrics['train_reg'].append(reg_err)
        self.metrics['val_reg'].append(val_reg_err)

        print(f"\n"
              f"Epoch {epoch:03d} | "
              f"Train Classification Accuracy: {self.metrics['train_class'][-1]:.4f} | "
              f"Validation Classification Accuracy: {self.metrics['val_class'][-1]:.4f} | "
              f"Train Regression Error: {reg_err:.4f} km | "
              f"Validation Regression Error: {val_reg_err:.4f} km", end='')

    def on_train_end(self, logs=None):
        epochs = np.arange(1, len(self.metrics['train_class']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_class'], 'o-', label='Train Accuracy')
        plt.plot(epochs, self.metrics['val_class'], 's--', label='Validation Accuracy')
        plt.title('Classification Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(SAVE_MODEL_ACCURACY_GRAPH_PATH(self.model_name))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_reg'], 'x-', label='Train Regression Error (km)')
        plt.plot(epochs, self.metrics['val_reg'], 'd--', label='Validation Regression Error (km)')
        plt.title('Regression Error per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Distance Error (km)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(SAVE_MODEL_REGRESSION_GRAPH_PATH(self.model_name))
        plt.close()


class HistorySaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.history = {}
        os.makedirs(os.path.dirname(HISTORY_PATH(model_name, 0)), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        for key, value in (logs or {}).items():
            self.history.setdefault(key, []).append(value)
        with open(HISTORY_PATH(self.model_name, epoch), 'wb') as f:
            pickle.dump(self.history, f)


def haversine_distance(observation_coordinates, prediction_coordinates, reduce_mean=True):
    observation_coordinates = tf.cast(observation_coordinates, tf.float32)
    prediction_coordinates = tf.cast(prediction_coordinates, tf.float32)

    degree_to_radian = tf.constant(0.017453292519943295, dtype=tf.float32)
    earth_radius_km = tf.constant(6378.1, dtype=tf.float32)

    observation_radians = observation_coordinates * degree_to_radian
    prediction_radians = prediction_coordinates * degree_to_radian

    observation_latitude = observation_radians[:, 0]
    observation_longitude = observation_radians[:, 1]
    prediction_latitude = prediction_radians[:, 0]
    prediction_longitude = prediction_radians[:, 1]

    latitude_delta = observation_latitude - prediction_latitude
    longitude_delta = observation_longitude - prediction_longitude

    latitude_sinus = tf.sin(latitude_delta / 2)
    longitude_sinus = tf.sin(longitude_delta / 2)

    a = latitude_sinus**2 + tf.cos(observation_latitude) * tf.cos(prediction_latitude) * longitude_sinus**2
    angular_distance_in_radians = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    km_distance = earth_radius_km * angular_distance_in_radians

    if reduce_mean: return tf.reduce_mean(km_distance)
    return km_distance


def enable_mixed_precision_if_supported():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled.")
        except Exception as e:
            print("Could not enable mixed precision:", e)
    else:
        print("No GPU detected, skipping mixed precision.")


haversine_distance_wrapper = tf.keras.metrics.MeanMetricWrapper(
    haversine_distance, name="haversine"
)
