from sklearn.model_selection import StratifiedShuffleSplit
from configs.settings import *
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np


def load_data_splits(data_path=DATA_IMAGES_PATH):
    class_folders = sorted([f for f in Path(data_path).iterdir() if f.is_dir()])
    image_paths, y_classification, y_regression = [], [], []
    num_classes = len(class_folders)

    for class_idx, class_folder in enumerate(class_folders):
        coordinates_folders = [f for f in class_folder.iterdir() if f.is_dir()]
        for coordinates_folder in coordinates_folders:
            images = list(coordinates_folder.glob("*.jpg"))
            if not images: continue
            one_hot = np.zeros(num_classes)
            one_hot[class_idx] = 1
            coordinates = np.array([float(c) for c in coordinates_folder.name.split(',')])

            for image_path in images:
                image_paths.append(str(image_path))
                y_classification.append(one_hot)
                y_regression.append(coordinates)

    y_classification = np.stack(y_classification)
    y_regression = np.stack(y_regression)

    with Image.open(image_paths[0]) as img:
        image_size = img.size[::-1]

    total_size = TEST_SIZE + VALIDATION_SIZE
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_size, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, y_classification.argmax(1)))

    rel_val_size = VALIDATION_SIZE / total_size
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=42)
    val_idx, test_idx = next(sss2.split([image_paths[i] for i in temp_idx], y_classification[temp_idx].argmax(1)))

    return {
        "train": ([image_paths[i] for i in train_idx],
                  y_classification[train_idx],
                  y_regression[train_idx]),
        "validation": ([image_paths[temp_idx[i]] for i in val_idx],
                       y_classification[temp_idx][val_idx],
                       y_regression[temp_idx][val_idx]),
        "test": ([image_paths[temp_idx[i]] for i in test_idx],
                 y_classification[temp_idx][test_idx],
                 y_regression[temp_idx][test_idx]),
        "image_shape": (image_size[0], image_size[1], 3)
    }


def make_dataset(image_paths, y_class, y_reg, img_size, batch_size=TRAIN_BATCH_SIZE):
    def _load_image(path, cls, reg):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, (cls, reg)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, y_class, y_reg))
    ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
