from src.data.utils import save_locations, chunk_list
from collections import defaultdict
from configs.settings import *
import shutil
import json

if __name__ == "__main__":
    with open(ALL_LOCATIONS_PATH) as file:
        grouped_locations = json.load(file)
        split_grouped_locations = defaultdict(dict)

        for group, locations in grouped_locations.items():
            split_locations = chunk_list(locations, SPLIT_LOCATIONS_PER_CELL)
            for split, locs in enumerate(split_locations):
                split_grouped_locations[split][group] = locs

        os.makedirs(SPLIT_FOLDER_PATH, exist_ok=True)
        shutil.rmtree(SPLIT_FOLDER_PATH)
        for split, grouped_locations in split_grouped_locations.items():
            save_locations(grouped_locations, f"{SPLIT_FOLDER_PATH}/split_{split}.json")
