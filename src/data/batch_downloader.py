from aiofiles import open as aio_open
from aiolimiter import AsyncLimiter
from configs.settings import *
import asyncio
import aiohttp
import json
import os

limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, 60)
stop_event = asyncio.Event()
log_lock = asyncio.Lock()


async def download_image(session, request, output_dir, log_path):
    async with limiter:
        if stop_event.is_set():
            return

        params = {
            "key": API_KEY,
            "location": request["location"],
            "heading": request["heading"],
            "pitch": request["pitch"],
            "fov": request["fov"],
            "size": request["size"]
        }
        try:
            async with session.get(STREETVIEW_API_URL, params=params) as response:
                if response.status == 200:
                    filename = f"{output_dir}/{request['cell']}/{request['location']}/{request['fov']}_{request['heading']}.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    async with aio_open(filename, "wb") as file:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            await file.write(chunk)

                    print(f"Downloaded image {request['request_id']}: {filename}")
                    async with log_lock:
                        async with aio_open(log_path, "a") as log_file:
                            await log_file.write(f"{request['request_id']} OK {filename}\n")
                else:
                    stop_event.set()
                    print(f"Failed request {request['request_id']}: {response.status}")
                    text = await response.text()
                    async with log_lock:
                        async with aio_open(log_path, "a") as log_file:
                            await log_file.write(f"{request.get('request_id')} ERROR {response.status} {text}\n")
        except Exception as e:
            stop_event.set()
            print(f"Error downloading request {request['request_id']}: {e}")
            async with log_lock:
                async with aio_open(log_path, "a") as log_file:
                    await log_file.write(f"{request.get('request_id')} EXCEPTION {str(e)}\n")


async def main():
    with open(DOWNLOAD_BATCH_FILE_PATH) as file:
        requests = json.load(file)

    batch_name = os.path.splitext(os.path.basename(DOWNLOAD_BATCH_FILE_PATH.rstrip('/\\')))[0]
    log_path = f"{DOWNLOAD_LOG_PATH}/download_{batch_name}.log"
    output_dir = f"{DOWNLOAD_FOLDER_PATH}/{requests[0]['total_cells']}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, request, output_dir, log_path) for request in requests]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
