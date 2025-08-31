from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Google Street View API parameters
API_KEY = os.getenv("GOOGLE_API_KEY")
STREETVIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"

# Parameters for generating the map
COUNTRIES_PATH = "data/external/countries.zip"
EUROPE_PATH = "data/intermediate/europe.gpkg"
GRID_PATH = "data/intermediate/grid.gpkg"
BOUNDARIES = (-25, 34, 35, 63)  # latitude and longitude degrees
CELL_SIZE = 180_000  # 180.000 m = 180 km
MIN_AREA = 9e9  # 11×10⁹ m² = 11_000 km²
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

# Parameters for splitting locations into defined number of samples per cell
SPLIT_FOLDER_PATH = "data/intermediate/split_locations"
SPLIT_LOCATIONS_PER_CELL = 100

# Parameters for preparations of download batches
DOWNLOAD_LOCATIONS_PATH = f"{SPLIT_FOLDER_PATH}/split_0.json"
DOWNLOAD_BATCHES_PATH = "data/intermediate/download_batches"
DOWNLOAD_IMAGES_PER_LOCATION = 5
DOWNLOAD_VERTICAL_FOV = 82.3187  # for height 640px, gives panorama wide 2300px, 79.9° would give 2400px wide panorama
DOWNLOAD_BATCH_SIZE = 10_000
DOWNLOAD_IMAGE_HEIGHT = 640

# Parameters for downloading single batch
DOWNLOAD_BATCH_FILE_PATH = f"{DOWNLOAD_BATCHES_PATH}/batch_0.json"
DOWNLOAD_LOG_PATH = f"data/intermediate"
DOWNLOAD_FOLDER_PATH = "data/raw"
MAX_REQUESTS_PER_MINUTE = 25000
CONCURRENT_DOWNLOADS = 1000
CHUNK_SIZE = 16 * 1024

# Parameters for plotting the map
PROJECTION_TYPE = "EPSG:3035"  # "EPSG:3035" preserves area, while "EPSG:4326" uses longitude and latitude
PLOT_LOCATIONS_PATH = ALL_LOCATIONS_PATH
ENUMERATE_CELLS = False
LOAD_LOCATIONS = True
FIG_SIZE = (9, 8)
