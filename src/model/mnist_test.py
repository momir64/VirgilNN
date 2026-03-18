from src.model.utils import AccuracyMeasurerCallback, HistorySaverCallback  # Custom callback
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from configs.settings import *
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

val_size = int(0.5 * len(x_test))
x_val, x_test_final = x_test[:val_size], x_test[val_size:]
y_val, y_test_final = y_test[:val_size], y_test[val_size:]

print("TensorFlow version: {}".format(tf.__version__))
print("Device name: {}".format(tf.test.gpu_device_name()))
print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

inputs = tf.keras.Input(shape=(784,), name='input')
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)

output1 = tf.keras.layers.Dense(10, activation='softmax', name='classification')(x)
output2 = tf.keras.layers.Dense(10, activation='softmax', name='coordinates')(x)

model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])

model.compile(
    optimizer='adam',
    loss={
        'classification': 'sparse_categorical_crossentropy',
        'coordinates': 'sparse_categorical_crossentropy'
    },
    metrics={
        'classification': 'accuracy',
        'coordinates': 'accuracy'
    }
)

print("Model output names:", model.output_names)

y_train_multi = {'classification': y_train, 'coordinates': y_train}
y_val_multi = {'classification': y_val, 'coordinates': y_val}
y_test_multi = {'classification': y_test_final, 'coordinates': y_test_final}

accuracy_measurer = AccuracyMeasurerCallback("mnist_test")
history_saver = HistorySaverCallback("mnist_test")
model_saver_frozen = ModelCheckpoint(MODEL_PATH("mnist_test" + "_frozen"), verbose=1, save_best_only=True)
model_saver_unfrozen = ModelCheckpoint(MODEL_PATH("mnist_test" + "_unfrozen"), verbose=1, save_best_only=True)

print('Training...')
epochs = 10
model.fit(
    x_train,
    y_train_multi,
    validation_data=(x_val, y_val_multi),
    epochs=1,
    batch_size=64,
    callbacks=[accuracy_measurer, history_saver, model_saver_frozen],
    verbose=0
)

model.compile(
optimizer='adam',
    loss={
        'classification': 'sparse_categorical_crossentropy',
        'coordinates': 'sparse_categorical_crossentropy'
    },
    metrics={
        'classification': 'accuracy',
        'coordinates': 'accuracy'
    }
)

model.fit(
    x_train,
    y_train_multi,
    validation_data=(x_val, y_val_multi),
    epochs=epochs - 1,
    batch_size=64,
    callbacks=[accuracy_measurer, history_saver, model_saver_unfrozen],
    verbose=0
)

print("\nLoading the last frozen model...")

print(f"Model with epoch {1 - 1:03d}")
model = load_model(MODEL_PATH("mnist_test" + "_frozen"))
model.evaluate(x_val, y_val_multi, batch_size=64)

print("\nLoading the last unfrozen model...")

print(f"Model with epoch {9 - 1:03d}")
model = load_model(MODEL_PATH("mnist_test" + "_unfrozen"))
model.evaluate(x_val, y_val_multi, batch_size=64)

# with open(HISTORY_PATH("mnist_test", epochs - 1), 'rb') as f:
#     last_epoch_history = pickle.load(f)
#     print(last_epoch_history)
