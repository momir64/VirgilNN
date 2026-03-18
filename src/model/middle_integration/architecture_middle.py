from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, Concatenate, LayerNormalization, Add, Conv2D, Multiply, GlobalAveragePooling2D, Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from configs.settings import UNFREEZE_LR
import src.model.utils as ut
import tensorflow as tf


@register_keras_serializable()
class ReduceSumStack(Layer):
    def call(self, inputs):
        return tf.reduce_sum(tf.stack(inputs, axis=1), axis=1)


def residual_block(x, units, dropout_rate=0.3):
    shortcut = x
    x = Dense(units)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units)(x)
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    x = Add()([x, shortcut])
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def deepsets_aggregation(features, embed_dim=128):
    phi_features = [Dense(embed_dim, activation='relu')(f) for f in features]
    x = ReduceSumStack()(phi_features)
    x = LayerNormalization()(x)
    x = Dense(embed_dim, activation='relu')(x)
    return x


def attention_pooling(x):
    attn = Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    x = Multiply()([x, attn])
    x = GlobalAveragePooling2D()(x)
    return x


def initialize_middle_model(input_shape, num_views, num_classes, pretrained_backbone_path=None, backbone_trainable=False, lr=8e-4):
    inputs = [Input(shape=input_shape) for _ in range(num_views)]
    shared_backbone = InceptionResNetV2(include_top=False, weights=None, input_shape=input_shape)

    if pretrained_backbone_path:
        print(f"Loading backbone weights from full model at: {pretrained_backbone_path}")
        full_model = load_model(pretrained_backbone_path, compile=False)

        for layer in shared_backbone.layers:
            try:
                pretrained_layer = full_model.get_layer(layer.name)
                if layer.weights and pretrained_layer.weights:
                    layer.set_weights(pretrained_layer.get_weights())
                    print(f"Loaded weights for layer: {layer.name}")
            except ValueError:
                print(f"Skipped layer (not found in pretrained model): {layer.name}")
            except Exception as e:
                print(f"Error loading weights for layer {layer.name}: {e}")

    shared_backbone.trainable = backbone_trainable

    features = [Dropout(0.2)(attention_pooling(shared_backbone(inp))) for inp in inputs]
    x = deepsets_aggregation(features, embed_dim=128)

    x_emb = Dense(128, activation='relu')(x)
    out_class_logits = Dense(num_classes, name='logits')(x_emb)
    out_class = tf.keras.layers.Softmax(name='classification')(out_class_logits)

    x_concat = Concatenate()([x_emb, out_class_logits])
    x_concat = residual_block(x_concat, 256)
    x_concat = residual_block(x_concat, 128)
    x_concat = Dense(64)(x_concat)
    x_concat = LeakyReLU(0.2)(x_concat)
    x_concat = Dropout(0.3)(x_concat)
    out_regression = Dense(2, activation='linear', name='coordinates')(x_concat)

    model = Model(inputs=inputs, outputs=[out_class, out_regression])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=['categorical_crossentropy', 'mae'],
        loss_weights=[1.0, 0.5],
        metrics={'classification': 'accuracy', 'coordinates': ut.haversine_distance_wrapper}
    )
    return model


def print_layer_status(model, indent=0):
    for layer in model.layers:
        prefix = "  " * indent
        print(f"{prefix}{layer.name}: {'Trainable' if layer.trainable else 'Frozen'}")
        if hasattr(layer, "layers"):
            print_layer_status(layer, indent + 2)


def initialize_unfrozen_model(model):
    for layer in model.layers:
        if 'inception_resnet_v2' in layer.name.lower():
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=UNFREEZE_LR, clipnorm=1.0),
        loss=['categorical_crossentropy', 'mae'],
        loss_weights=[1.0, 0.5],
        metrics={'classification': 'accuracy', 'coordinates': ut.haversine_distance_wrapper}
    )

    print_layer_status(model)
    return model
