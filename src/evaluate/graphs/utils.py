from src.model.middle_integration.data_loader_middle import load_data_splits_grouped, make_grouped_dataset
from src.model.middle_integration.architecture_middle import ReduceSumStack
from src.model.data_loader_single import load_data_splits, make_dataset
from tensorflow.keras.models import load_model
from configs.settings import *
from keras import config


def load_test_data(model_name, batch_size=128):
    if "middle" in model_name:
        data_splits = load_data_splits_grouped(EVALUATION_GROUPED_IMAGES_PATH)
        image_shape = data_splits["image_shape"]
        test_ds = make_grouped_dataset(*data_splits["test"], image_shape[:2], batch_size)
    else:
        data_splits = load_data_splits(EVALUATION_IMAGES_PATH)
        image_shape = data_splits["image_shape"]
        test_ds = make_dataset(*data_splits["test"], image_shape[:2], batch_size)

    config.enable_unsafe_deserialization()
    model = load_model(MODEL_PATH(model_name), compile=False, custom_objects={"ReduceSumStack": ReduceSumStack})
    return test_ds, model
