from configs.settings import NON_SCALED_PATH, SCALED_PATH, SCALING_FACTOR
import concurrent.futures
from pathlib import Path
import numpy as np
import asyncio
import cv2
import os
import re


def numeric_sort_key(filename: str) -> float:
    parts = re.findall(r'\d+(?:\.\d+)?', Path(filename).stem)
    return float(parts[-1]) if parts else 0.0


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    if new_width == 0 or new_height == 0:
        raise ValueError(f"Scaled dimensions too small ({new_width}x{new_height}).")
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def process_subfolder(main_idx: int, total: int, main_path: Path, output_main: Path, subfolder: str,
                      scale_factor: float) -> str:
    sub_path = main_path / subfolder
    image_files = sorted(
        [f for f in os.listdir(sub_path) if f.lower().endswith(".jpg")],
        key=numeric_sort_key
    )
    if not image_files:
        return f"[{main_idx}/{total}] Skipping {sub_path}, no images found."

    output_sub_path = output_main / subfolder
    output_sub_path.mkdir(parents=True, exist_ok=True)

    for filename in image_files:
        img_path = sub_path / filename
        output_path = output_sub_path / filename
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read {img_path}")

        scaled = scale_image(img, scale_factor)
        cv2.imwrite(str(output_path), scaled)

    return f"[{main_idx}/{total}] Scaled images saved to: {output_sub_path}"


async def main() -> None:
    raw_root = Path(NON_SCALED_PATH)
    scaled_root = Path(SCALED_PATH)
    folders = [f for f in sorted(raw_root.iterdir()) if f.is_dir()]
    total = len(folders)
    print(f"Found {total} main folders.")

    tasks = []
    processed = 0
    loop = asyncio.get_running_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as pool:
        for i, main_folder in enumerate(folders, start=1):
            output_main = scaled_root / main_folder.name
            output_main.mkdir(parents=True, exist_ok=True)
            for subfolder in [sf.name for sf in main_folder.iterdir() if sf.is_dir()]:
                tasks.append(
                    loop.run_in_executor(pool, process_subfolder, i, total, main_folder, output_main, subfolder,
                                         SCALING_FACTOR))

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                processed += 1
                print(result)
            except Exception as e:
                print(f"Error: {e}")

    print(f"Completed scaling for {processed} subfolders across {total} main folders.")


if __name__ == "__main__":
    asyncio.run(main())
