from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")

# Map generation parameters
COUNTRIES_PATH = "data/external/countries.zip"
EUROPE_PATH = "data/intermediate/europe.gpkg"
GRID_PATH = "data/intermediate/grid.gpkg"
BOUNDARIES = (-25, 34, 35, 63.5)  # latitude and longitude degrees
GRID_SIZE = 200_000  # 200.000 m = 200 km
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

# Parameter for
SPLIT_FOLDER_PATH = "data/intermediate/split_locations"
SPLIT_LOCATIONS_PER_CELL = 150
