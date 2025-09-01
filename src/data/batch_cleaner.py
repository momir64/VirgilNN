from configs.settings import *
from src.data.utils import *
import json


def main():
    ok_ids = set()
    with open(RECOVER_LOG_PATH, "r") as file:
        for line in file:
            parts = line.split()
            if parts[1] == "OK":
                ok_ids.add(int(parts[0]))

    with open(RECOVER_BATCH_PATH, "r") as file:
        data = json.load(file)

    filtered_requests = [entry for entry in data if entry["request_id"] not in ok_ids]
    batch_name = os.path.splitext(os.path.basename(RECOVER_BATCH_PATH.rstrip('/\\')))[0]
    save_batch(filtered_requests, f"{DOWNLOAD_BATCHES_PATH}/recover_{batch_name}.json")


if __name__ == "__main__":
    main()
