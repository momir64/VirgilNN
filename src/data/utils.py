from typing import Any
import json
import os


def chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def save_locations(locations_per_cell: dict[str, dict[str, Any]], file_path: str) -> None:
    sorted_locations = dict(sorted(locations_per_cell.items(), key=lambda item: int(item[0])))
    json_str = json.dumps(sorted_locations, indent=2)
    json_str = json_str.replace("\n    {\n      ", "\n    {")
    json_str = json_str.replace("\n    }", "}")
    json_str = json_str.replace(",\n      ", ", ")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(json_str)

def save_batch(requests: list[dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w") as file:
        json_str = json.dumps(requests)
        json_str = json_str.replace("}, {", "},\n  {")
        json_str = json_str.replace("[", "[\n  ")
        json_str = json_str.replace("]", "  \n]")
        file.write(json_str)