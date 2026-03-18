from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Concatenate, Dropout, LeakyReLU
from tensorflow.keras import Model, Input
import src.model.utils as ut
import tensorflow as tf


def initialize_model(input_shape, number_of_classes):
    print('Initialising Network')

    classification_out_shape = number_of_classes
    regression_out_shape = 2

    input_layer = Input(shape=input_shape)

    feature_extraction = InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        input_tensor=input_layer
    )

    embed = feature_extraction.output

    out_classification = Dense(classification_out_shape, activation='softmax', name='classification')(embed)

    concat = Concatenate()([embed, out_classification])
    dense_1 = Dense(250, name='mid_layer')(concat)
    drop = Dropout(0.5)(dense_1)
    dense_2 = Dense(200, name='mid_layer_1')(drop)
    dense_3 = Dense(150, name='mid_layer_2')(dense_2)

    dense_3 = LeakyReLU(alpha=0.25)(dense_3)  # potentially optional

    dense_4 = Dense(100, kernel_regularizer='l2', name='mid_layer_3')(dense_3)
    dense_5 = Dense(80, kernel_regularizer='l2', name='mid_layer_4')(dense_4)
    drop = Dropout(0.5)(dense_5)
    dense_6 = Dense(50, kernel_regularizer='l2', name='mid_layer_5')(drop)

    out_regression = Dense(regression_out_shape, activation='linear', name='coordinates')(dense_6)

    model = Model(inputs=input_layer, outputs=[out_classification, out_regression])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss=['categorical_crossentropy', 'mae'],
        metrics={'classification':'accuracy', 'coordinates': ut.haversine_distance_wrapper}
    )
    model.summary()

    print(f"Network Initialised and compiled. "
          f"Input shape: {model.input_shape}, "
          f"Output shape: {model.output_shape}")

    return model
