from configs.settings import *
from pathlib import Path
from PIL import Image
import asyncio

async def mirror_image(image_path: Path, input_root: Path, output_root: Path,
                       subfolder_count: int, total_subfolders: int, image_count: int):
    def task():
        img = Image.open(image_path)
        rel_path = image_path.relative_to(input_root)
        subfolder = rel_path.parent
        original_path = output_root / subfolder / f"{image_path.stem}_original{image_path.suffix}"
        mirrored_path = output_root / subfolder / f"{image_path.stem}_mirrored{image_path.suffix}"
        mirrored_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(original_path)
        mirrored = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mirrored.save(mirrored_path)
        return f"[{subfolder_count}/{total_subfolders}] ({image_count}) {image_path.name}"

    try:
        msg = await asyncio.to_thread(task)
        print(msg)
    except Exception as e:
        print(f"{image_path}: {e}")

def find_images_in_subfolders(root: Path):
    subfolders = sorted([s for s in root.iterdir() if s.is_dir()],
                        key=lambda x: int(x.name) if x.name.isdigit() else x.name)
    images_by_subfolder = []
    for idx, sub in enumerate(subfolders, start=1):
        imgs = sorted(sub.rglob('*.jpg'))
        images_by_subfolder.append((idx, imgs))
    return images_by_subfolder

async def main():
    input_root = Path(NON_AUGMENTED_IMAGES_PATH)
    if not input_root.exists():
        print(f"Input folder not found: {input_root}")
        return
    output_root = Path(AUGMENTED_IMAGES_PATH) / input_root.name

    images_by_subfolder = find_images_in_subfolders(input_root)
    total_subfolders = len(images_by_subfolder)
    if total_subfolders == 0:
        print("No subfolders found. Exiting.")
        return

    semaphore = asyncio.Semaphore(64)

    async def sem_task(coro):
        async with semaphore:
            await coro

    tasks = []
    for subfolder_count, imgs in images_by_subfolder:
        for image_count, img_path in enumerate(imgs, start=1):
            tasks.append(sem_task(mirror_image(img_path, input_root, output_root,
                                               subfolder_count, total_subfolders, image_count)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
