from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Concatenate, Dropout, LeakyReLU
from tensorflow.keras import Model, Input
import src.model.utils as ut
import tensorflow as tf


def initialize_model(input_shape, num_classes, loss_weights=None):
    print('Initialising Network')

    if loss_weights is None:
        loss_weights = [1.0, 0.5]

    input_layer = Input(shape=input_shape)
    backbone = InceptionResNetV2(
        include_top=False,
        input_tensor=input_layer,
        pooling='avg',
        input_shape=input_shape
    )
    embed = backbone.output

    out_class_logits = Dense(num_classes, name='logits')(embed)
    out_class = tf.keras.layers.Softmax(name='classification')(out_class_logits)

    x = Concatenate()([embed, out_class_logits])
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer='l2')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, kernel_regularizer='l2')(x)
    x = LeakyReLU(alpha=0.2)(x)
    out_regression = Dense(2, activation='linear', name='coordinates')(x)

    model = Model(inputs=input_layer, outputs=[out_class, out_regression])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss=['categorical_crossentropy', 'mae'],
        loss_weights=loss_weights,
        metrics={'classification':'accuracy', 'coordinates': ut.haversine_distance_wrapper}
    )

    model.summary()

    print(f"Network Initialised and compiled. "
          f"Input shape: {model.input_shape}, "
          f"Output shape: {model.output_shape}")

    return model

