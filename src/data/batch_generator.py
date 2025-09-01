from configs.settings import *
from src.data.utils import *
import urllib.parse
import hashlib
import shutil
import random
import base64
import hmac
import json
import math


def generate_signature(parameters):
    keys = ["location", "heading", "pitch", "fov", "size"]
    query_parameters = {"key": API_KEY, **{key: parameters[key] for key in keys}}
    query_string = urllib.parse.urlencode(query_parameters).replace('%2C', ',')
    decoded_secret = base64.urlsafe_b64decode(API_SECRET)
    url_to_sign = f"{urllib.parse.urlparse(STREETVIEW_API_URL).path}?{query_string}"
    signature = hmac.new(decoded_secret, url_to_sign.encode(), hashlib.sha1)
    return base64.urlsafe_b64encode(signature.digest()).decode()


def main():
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
                    parameters = {
                        "request_id": len(download_items),
                        "total_cells": len(grouped_locations),
                        "cell": group,
                        "size": f"{width}x{DOWNLOAD_IMAGE_HEIGHT}",
                        "fov": horizontal_fov,
                        "pitch": 0,
                        "location": f"{location["lat"]},{location["lng"]}",
                        "heading": heading
                    }
                    parameters["signature"] = generate_signature(parameters)
                    download_items.append(parameters)

    os.makedirs(DOWNLOAD_BATCHES_PATH, exist_ok=True)
    shutil.rmtree(DOWNLOAD_BATCHES_PATH)
    os.makedirs(DOWNLOAD_BATCHES_PATH, exist_ok=True)
    for i, batch in enumerate(chunk_list(download_items, DOWNLOAD_BATCH_SIZE)):
        file_path = f"{DOWNLOAD_BATCHES_PATH}/batch_{i}.json"
        save_batch(batch, file_path)


if __name__ == "__main__":
    main()
