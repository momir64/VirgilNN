from sklearn.model_selection import StratifiedShuffleSplit
from configs.settings import *
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np

def load_data_splits_grouped(data_path=DATA_IMAGES_PATH):
    class_folders = sorted([f for f in Path(data_path).iterdir() if f.is_dir()])
    grouped_paths, y_classification, y_regression = [], [], []
    num_classes = len(class_folders)

    views_per_sample = None

    for class_idx, class_folder in enumerate(class_folders):
        coords_folders = [f for f in class_folder.iterdir() if f.is_dir()]
        for coords in coords_folders:
            all_images = sorted([p for p in coords.glob("*.jpg")])
            if not all_images: continue

            originals = [p for p in all_images if "_original" in p.stem]
            mirrored = [p for p in all_images if "_mirrored" in p.stem]
            min_views = min(len(originals), len(mirrored))
            if min_views == 0:
                continue

            if views_per_sample is None:
                views_per_sample = min_views
            else:
                views_per_sample = min(views_per_sample, min_views)

            originals = sorted(originals)[:views_per_sample]
            mirrored = sorted(mirrored)[:views_per_sample]

            one_hot = np.zeros(num_classes); one_hot[class_idx] = 1
            coords_vec = np.array([float(c) for c in coords.name.split(',')])

            grouped_paths.append([str(p) for p in originals])
            grouped_paths.append([str(p) for p in mirrored])
            y_classification.append(one_hot)
            y_classification.append(one_hot)
            y_regression.append(coords_vec)
            y_regression.append(coords_vec)

    y_classification = np.stack(y_classification)
    y_regression = np.stack(y_regression)

    with Image.open(grouped_paths[0][0]) as img:
        image_size = img.size[::-1]

    total_size = TEST_SIZE + VALIDATION_SIZE
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_size, random_state=42)
    labels = y_classification.argmax(1)
    indices = np.arange(len(grouped_paths))
    train_idx, temp_idx = next(sss1.split(indices, labels))

    rel_val_size = VALIDATION_SIZE / total_size
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=42)
    val_idx, test_idx = next(sss2.split(indices[temp_idx], labels[temp_idx]))

    return {
        "train": ([grouped_paths[i] for i in train_idx],
                  y_classification[train_idx],
                  y_regression[train_idx]),
        "validation": ([grouped_paths[temp_idx[i]] for i in val_idx],
                       y_classification[temp_idx][val_idx],
                       y_regression[temp_idx][val_idx]),
        "test": ([grouped_paths[temp_idx[i]] for i in test_idx],
                 y_classification[temp_idx][test_idx],
                 y_regression[temp_idx][test_idx]),
        "image_shape": (image_size[0], image_size[1], 3),
        "views_per_sample": views_per_sample
    }


def make_grouped_dataset(image_group_paths, y_class, y_reg, img_size, batch_size=TRAIN_BATCH_SIZE):
    def process_group(paths, cls, reg):
        def load_img(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img

        imgs = tf.map_fn(load_img, paths, dtype=tf.float32)
        return imgs, (cls, reg)

    def split_views(batch_x, batch_y):
        views = tf.unstack(batch_x, axis=1)
        return tuple(views), batch_y

    flat_paths = []
    labels_cls = []
    labels_reg = []

    for paths, cls, reg in zip(image_group_paths, y_class, y_reg):
        flat_paths.append(paths)
        labels_cls.append(cls)
        labels_reg.append(reg)

    flat_paths = np.array(flat_paths)
    labels_cls = np.array(labels_cls)
    labels_reg = np.array(labels_reg)

    ds = tf.data.Dataset.from_tensor_slices((flat_paths, labels_cls, labels_reg))
    ds = ds.shuffle(buffer_size=len(flat_paths))
    ds = ds.map(process_group, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(split_views, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds