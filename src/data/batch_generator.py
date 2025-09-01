from configs.settings import *
from src.data.utils import *
import shutil
import random
import json
import math

if __name__ == "__main__":
    with open(DOWNLOAD_LOCATIONS_PATH) as file:
        grouped_locations = json.load(file)
        vertical_fov = math.radians(DOWNLOAD_VERTICAL_FOV)
        horizontal_fov = round(360 / DOWNLOAD_IMAGES_PER_LOCATION)
        width = round(DOWNLOAD_IMAGE_HEIGHT * math.pi / math.tan(vertical_fov / 2) / DOWNLOAD_IMAGES_PER_LOCATION)
        download_items = []

        for group, locations in grouped_locations.items():
            for location in locations:
                random_heading_offset = random.uniform(0, 360)
                for i in range(DOWNLOAD_IMAGES_PER_LOCATION):
                    heading = (i * horizontal_fov + horizontal_fov / 2 + random_heading_offset) % 360
                    download_items.append({
                        "request_id": len(download_items),
                        "total_cells": len(grouped_locations),
                        "cell": group,
                        "size": f"{width}x{DOWNLOAD_IMAGE_HEIGHT}",
                        "fov": horizontal_fov,
                        "pitch": 0,
                        "location": f"{location["lat"]},{location["lng"]}",
                        "heading": heading
                    })

    os.makedirs(DOWNLOAD_BATCHES_PATH, exist_ok=True)
    shutil.rmtree(DOWNLOAD_BATCHES_PATH)
    os.makedirs(DOWNLOAD_BATCHES_PATH, exist_ok=True)
    for i, batch in enumerate(chunk_list(download_items, DOWNLOAD_BATCH_SIZE)):
        file_path = f"{DOWNLOAD_BATCHES_PATH}/batch_{i}.json"
        save_batch(batch, file_path)
