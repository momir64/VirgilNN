from configs.settings import *
import concurrent.futures
from pathlib import Path
from typing import List
import numpy as np
import asyncio
import cv2
import os
import re

def numeric_sort_key(filename: str) -> float:
    parts = re.findall(r'\d+(?:\.\d+)?', Path(filename).stem)
    return float(parts[-1]) if parts else 0.0

def create_panorama(images: List[np.ndarray]) -> np.ndarray:
    min_height = min(img.shape[0] for img in images)
    resized = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
    return np.hstack(resized)

def process_subfolder(main_idx: int, total: int, main_path: Path, output_main: Path, subfolder: str) -> str:
    sub_path = main_path / subfolder
    image_files = sorted(
        [f for f in os.listdir(sub_path) if f.lower().endswith(".jpg")],
        key=numeric_sort_key
    )
    if len(image_files) < 2:
        return f"[{main_idx}/{total}] Skipping {sub_path}, not enough images."

    images = []
    for filename in image_files:
        img_path = sub_path / filename
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read {img_path}")
        images.append(img)

    panorama = create_panorama(images)
    output_subfolder = output_main / subfolder
    output_subfolder.mkdir(parents=True, exist_ok=True)
    output_path = output_subfolder / "panorama.jpg"
    cv2.imwrite(str(output_path), panorama)
    return f"[{main_idx}/{total}] Saved panorama: {output_path}"

async def main() -> None:
    raw_root = Path(RAW_IMAGES_PATH)
    pano_root = Path(PANORAMAS_PATH)
    folders = [f for f in sorted(raw_root.iterdir()) if f.is_dir()]
    total = len(folders)
    print(f"Found {total} main folders.")

    tasks = []
    processed = 0
    loop = asyncio.get_running_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as pool:
        for i, main_folder in enumerate(folders, start=1):
            output_main = pano_root / main_folder.name
            output_main.mkdir(parents=True, exist_ok=True)
            for subfolder in [sf.name for sf in main_folder.iterdir() if sf.is_dir()]:
                tasks.append(loop.run_in_executor(pool, process_subfolder, i, total, main_folder, output_main, subfolder))

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                processed += 1
                print(result)
            except Exception as e:
                print(f"Error: {e}")

    print(f"Completed {processed} panoramas out of {total} folders.")

if __name__ == "__main__":
    asyncio.run(main())
