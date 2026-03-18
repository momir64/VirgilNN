from src.model.middle_integration.architecture_middle import ReduceSumStack
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import tensorflow as tf
from glob import glob
from PIL import Image
import numpy as np
import cv2
import os


def reverse_lex_order_softmax_tensor(class_pred):
    if class_pred.dtype != tf.float32:
        class_pred = tf.cast(class_pred, tf.float32)
    num_classes = class_pred.shape[1]
    lex_order = sorted(range(num_classes), key=lambda x: str(x))
    reverse_map = [lex_order.index(i) for i in range(num_classes)]
    return tf.gather(class_pred, reverse_map, axis=1)


def reverse_lex_order_softmax(class_pred):
    num_classes = len(class_pred)
    lex_order = sorted(range(num_classes), key=lambda x: str(x))
    return {i: class_pred[lex_order.index(i)] for i in range(num_classes)}


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")


def _compute_gradcam(model, inputs, class_idx, layer_name=None):
    heatmaps = []
    if layer_name is None:
        conv_layer = model.get_layer(get_last_conv_layer(model))
    else:
        conv_layer = model.get_layer(layer_name)

    for _ in range(len(inputs[0]) if isinstance(inputs, list) else 1):
        if isinstance(model.input, list):
            grad_inputs = inputs
        else:
            grad_inputs = inputs[0]

        grad_model = Model(model.inputs, [conv_layer.output, model.outputs[0]])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(grad_inputs)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        heatmaps.append(heatmap.numpy())
    return heatmaps


def _prepare_image(image_path, target_size):
    width, height = map(int, target_size)
    img = Image.open(image_path).convert('RGB').resize((width, height))
    return np.array(img, dtype=np.float32) / 255.0


def _predict_from_model(model, inputs):
    predicts = model.predict(inputs)
    if isinstance(predicts, list) and len(predicts) >= 2:
        class_pred, coords_pred = predicts[0][0], predicts[1][0]
    else:
        raise ValueError("Model must output [class_pred, coords_pred]")
    softmax_dict = reverse_lex_order_softmax(class_pred)
    return tuple(coords_pred), softmax_dict


def predict_panorama(model_path, image_path):
    model = load_model(model_path, compile=False)
    img_height, img_width = model.input_shape[1:3]
    img_array = [np.expand_dims(_prepare_image(image_path, (img_width, img_height)), axis=0)]

    coords, softmax_dict = _predict_from_model(model, [img_array[0]])
    heatmap = _compute_gradcam(model, img_array, np.argmax(list(softmax_dict.values())))[0]
    heatmap = cv2.resize(heatmap, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return coords, softmax_dict, heatmap


def predict_panorama_group(model_path, folder_path):
    model = load_model(model_path, compile=False, custom_objects={"ReduceSumStack": ReduceSumStack})
    num_inputs = len(model.inputs)

    input_shapes = []
    if isinstance(model.input_shape, list):
        for shape in model.input_shape:
            input_shapes.append((shape[1] or 224, shape[2] or 224))
    else:
        shape = model.input_shape
        input_shapes.append((shape[1] or 224, shape[2] or 224))

    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))[:num_inputs]
    if len(image_paths) < num_inputs:
        raise ValueError(f"Expected at least {num_inputs} slices, found {len(image_paths)}")

    img_arrays = []
    for p, (h, w) in zip(image_paths, input_shapes):
        img_arrays.append(_prepare_image(p, (w, h)))

    if isinstance(model.input, list):
        model_inputs = [np.expand_dims(arr, axis=0) for arr in img_arrays]
    else:
        model_inputs = np.expand_dims(img_arrays[0], axis=0)

    coords, softmax_dict = _predict_from_model(model, model_inputs)

    return coords, softmax_dict
