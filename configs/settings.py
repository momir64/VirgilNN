from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")
STREETVIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"

# Map generation parameters
COUNTRIES_PATH = "data/external/countries.zip"
EUROPE_PATH = "data/intermediate/europe.gpkg"
GRID_PATH = "data/intermediate/grid.gpkg"
BOUNDARIES = (-25, 34, 35, 63.5)  # latitude and longitude degrees
CELL_SIZE = 200_000  # 200.000 m = 200 km
MIN_AREA = 1e10  # 1×10¹⁰ m² = 10_000 km²
EXCLUDE_COUNTRIES = [
    "Belarus", "Bulgaria", "Estonia", "Finland",
    "Iceland", "Latvia", "Lithuania", "Moldova",
    "Poland", "Romania", "Russia", "Ukraine"
]

# Location generation parameters
ALL_LOCATIONS_PATH = "data/intermediate/all_locations.json"
REQUESTS_PER_SECOND = 500
MAX_CONCURRENT_CELLS = 5
LOCATIONS_PER_CELL = 300
RADIUS = 1000  # 1 km

# Parameters for splitting locations
SPLIT_FOLDER_PATH = "data/intermediate/split_locations"
SPLIT_LOCATIONS_PER_CELL = 100

# Parameters for preparations of download batches
DOWNLOAD_LOCATIONS_PATH = f"{SPLIT_FOLDER_PATH}/split_0.json"
DOWNLOAD_BATCHES_PATH = "data/intermediate/download_batches"
DOWNLOAD_IMAGES_PER_LOCATION = 5
DOWNLOAD_VERTICAL_FOV = 84.85  # gives panorama 2200px wide, 79.9 would give 2400px wide panorama
DOWNLOAD_BATCH_SIZE = 10_000
DOWNLOAD_IMAGE_HEIGHT = 640

# Batch download parameters
DOWNLOAD_BATCH_FILE_PATH = f"{DOWNLOAD_BATCHES_PATH}/batch_0.json"
DOWNLOAD_LOG_PATH = f"data/intermediate"
DOWNLOAD_FOLDER_PATH = "data/raw"
MAX_REQUESTS_PER_MINUTE = 25000
CONCURRENT_DOWNLOADS = 1000
CHUNK_SIZE = 16 * 1024