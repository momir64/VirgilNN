from dotenv import load_dotenv
from pathlib import Path
import os

# Remove warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Google Street View API parameters
API_KEY = os.getenv("GOOGLE_API_KEY")
API_SECRET = os.getenv("GOOGLE_API_SECRET")
STREETVIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"

# Parameters for generating the map
COUNTRIES_PATH = "data/external/countries.zip"
EUROPE_PATH = "data/intermediate/europe.gpkg"
GRID_PATH = "data/intermediate/grid.gpkg"
BOUNDARIES = (-25, 34, 35, 63)  # latitude and longitude degrees
CELL_SIZE = 180_000  # 180.000 m = 180 km
MIN_AREA = 9e9  # 9×10⁹ m² = 9_000 km²
GRID_OFFSET_X = 20_000
GRID_OFFSET_Y = 20_000
EXCLUDE_COUNTRIES = [
    "Albania", "Belarus", "Bosnia and Herz.", "Bulgaria",
    "Estonia", "Finland", "Greece", "Iceland", "Latvia",
    "Lithuania", "Moldova", "Montenegro", "North Macedonia",
    "Poland", "Romania", "Russia", "Serbia", "Ukraine"
]

# Parameters for generating locations
ALL_LOCATIONS_PATH = "data/intermediate/all_locations.json"
USE_ONLY_MADE_BY_GOOGLE = True
REQUESTS_PER_SECOND = 500
MAX_CONCURRENT_CELLS = 5
LOCATIONS_PER_CELL = 300
RADIUS = 1000  # 1 km

# Parameters for splitting locations into a defined number of samples per cell
SPLIT_FOLDER_PATH = "data/intermediate/split_locations"
SPLIT_LOCATIONS_PER_CELL = 100

# Parameters for preparations of download batches
DOWNLOAD_LOCATIONS_PATH = f"{SPLIT_FOLDER_PATH}/split_1.json"
DOWNLOAD_BATCHES_PATH = "data/intermediate/download_batches"
DOWNLOAD_IMAGES_PER_LOCATION = 5
DOWNLOAD_VERTICAL_FOV = 82.3187  # for height 640px, gives panorama wide 2300px; 79.9° would give 2400px wide panorama
DOWNLOAD_BATCH_SIZE = 10_000
DOWNLOAD_IMAGE_HEIGHT = 640

# Parameters for downloading a single batch
DOWNLOAD_BATCH_FILE_PATH = f"{DOWNLOAD_BATCHES_PATH}/batch_0.json"
DOWNLOAD_LOG_PATH = f"data/intermediate/logs"
DOWNLOAD_FOLDER_PATH = "data/raw"
MAX_REQUESTS_PER_MINUTE = 25000
CONCURRENT_DOWNLOADS = 1000
CHUNK_SIZE = 16 * 1024

# Parameters for cleaning the batch file using a log file
RECOVER_BATCH_PATH = f"{DOWNLOAD_BATCHES_PATH}/batch_4.json"
RECOVER_LOG_PATH = f"{DOWNLOAD_LOG_PATH}/download_batch_4.log"

# Parameters for plotting the map
PROJECTION_TYPE = "EPSG:3035"  # "EPSG:3035" preserves area, while "EPSG:4326" uses longitude and latitude
PLOT_LOCATIONS_PATH = ALL_LOCATIONS_PATH
KEEP_ASPECT_RATIO = True
ENUMERATE_CELLS = False
LOAD_LOCATIONS = True
PLOT_GRIDS = True
FIG_SIZE = (8, 8)

# Parameters for image scaling
NON_SCALED_PATH = "data/raw/108"
SCALED_PATH = "data/processed/scaled"
SCALING_FACTOR = 0.5

# Parameters for creating panoramas
RAW_IMAGES_PATH = "data/processed/scaled"
PANORAMAS_PATH = "data/processed/panoramas"

# Parameters for image augmentation
NON_AUGMENTED_IMAGES_PATH = "data/processed/panoramas"
AUGMENTED_IMAGES_PATH = "data/processed/augmented"

# Pararameters for data loading and spliting
DATA_IMAGES_PATH = "data/processed/augmented/scaled"
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

# Parameters for general training
TRAIN_BATCH_SIZE = 8
USE_MIXED_PRECISION = True

def MODEL_PATH(model_name):
    return f"models/{model_name}/model/{model_name}.keras"

def HISTORY_PATH(model_name, epoch):
    return f"models/{model_name}/histories/history_{model_name}_epoch{epoch:03d}.pkl"

def SAVE_MODEL_ACCURACY_GRAPH_PATH(model_name):
    return f"models/{model_name}/graphs/accuracy_{model_name}.png"

def SAVE_MODEL_REGRESSION_GRAPH_PATH(model_name):
    return f"models/{model_name}/graphs/regression_{model_name}.png"

# Parameters for early integration testing
EPOCHS_EARLY = 100

# Parameters for middle integration training
BACKBONE_TRAINABLE = False
EPOCHS_SINGLE = 75
EPOCHS_MIDDLE = 60
UNFREEZE_EPOCH = 70
UNFREEZE_LR = 8e-5

# Parameters for GUI
GUI_IMAGES_PATH = "data/evaluation/images"
GUI_PANORAMA_PATH = "data/evaluation"

# Parameters for evaluation plots
EVALUATION_GROUPED_IMAGES_PATH = "data/processed/augmented/scaled"
EVALUATION_IMAGES_PATH = "data/processed/augmented/panoramas"
SAVE_GRAPH_PATH = "data/evaluation/graphs"
