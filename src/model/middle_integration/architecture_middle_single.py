from tensorflow.keras.layers import Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def build_single_view_model(input_shape, num_classes, lr=1e-4):
    base = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    class_out = Dense(num_classes, activation="softmax", name="classification")(x)
    x = Concatenate()([x, class_out])
    reg_out = Dense(2, name="coordinates")(x)
    model = Model(inputs=base.input, outputs=[class_out, reg_out])
    model.compile(
        optimizer=Adam(lr),
        loss={
            "classification": "categorical_crossentropy",
            "coordinates": "mse"
        },
        loss_weights={"classification": 1.0, "coordinates": 0.2},
        metrics={"classification": "accuracy", "coordinates": "mae"}
    )
    return model
