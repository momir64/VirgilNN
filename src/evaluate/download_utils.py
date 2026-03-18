import asyncio, aiohttp, urllib.parse, hashlib, base64, hmac, math
from aiofiles import open as aio_open
from aiolimiter import AsyncLimiter
from configs.settings import *
import random


def generate_signed_params(lat, lng, heading, pitch, fov, size):
    params = {
        "key": API_KEY,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "size": size
    }
    query_string = urllib.parse.urlencode(params).replace('%2C', ',')
    url_to_sign = f"{urllib.parse.urlparse(STREETVIEW_API_URL).path}?{query_string}"
    decoded_secret = base64.urlsafe_b64decode(API_SECRET)
    signature = hmac.new(decoded_secret, url_to_sign.encode(), hashlib.sha1)
    params["signature"] = base64.urlsafe_b64encode(signature.digest()).decode()
    return params


async def get_valid_google_coords(session, lat, lng, only_google, retries=10, radius=250, max_radius=16000):
    url = lambda r: f"{STREETVIEW_API_URL}/metadata?key={API_KEY}&radius={r}&location={lat},{lng}"
    for attempt in range(retries):
        try:
            async with session.get(url(radius), timeout=10) as resp:
                data = await resp.json()
            status = data.get("status", "UNKNOWN")
            if status == "OK" and "location" in data:
                loc = data["location"]
                if not only_google or data.get("copyright", "") == "© Google":
                    return loc["lat"], loc["lng"]
                else:
                    radius = int(max(250.0, radius / 2.5))
                    lat += random.uniform(-0.01, 0.01)
                    lng += random.uniform(-0.01, 0.01)
                    print("NOT GOOGLE")
            elif status == "OVER_QUERY_LIMIT":
                await asyncio.sleep(2 ** attempt)
            else:
                radius = int(min(radius * 2, max_radius))
                print(f"{status} new radius: {radius}")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt)
    return None, None


async def download_image(session, lat, lng, output_dir, images_per_location=DOWNLOAD_IMAGES_PER_LOCATION):
    output_dir = f"{output_dir}/{round(float(lat), 12)},{round(float(lng), 12)}/sliced"
    os.makedirs(output_dir, exist_ok=True)
    limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, 60)
    vertical_fov = math.radians(DOWNLOAD_VERTICAL_FOV)
    horizontal_fov = round(360 / images_per_location)
    width = round(DOWNLOAD_IMAGE_HEIGHT * math.pi / math.tan(vertical_fov / 2) / images_per_location)
    size = f"{width}x{DOWNLOAD_IMAGE_HEIGHT}"

    for i in range(images_per_location):
        async with limiter:
            heading = (i * horizontal_fov + horizontal_fov / 2) % 360
            params = generate_signed_params(lat, lng, heading, 0, horizontal_fov, size)
            filename = f"{output_dir}/{int(heading)}.jpg"
            try:
                async with session.get(STREETVIEW_API_URL, params=params) as response:
                    if response.status == 200:
                        async with aio_open(filename, "wb") as f:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                await f.write(chunk)
                        print(f"Downloaded {filename}")
                    else:
                        print(f"Failed to download ({lat},{lng}) heading {heading}, status {response.status}")
            except Exception as e:
                print(f"Error downloading heading {heading}: {e}")


def download_location(lat, lng, output_dir, only_google):
    async def wrapper():
        async with aiohttp.ClientSession() as session:
            real_lat, real_lng = await get_valid_google_coords(session, lat, lng, only_google)
            if real_lat is None:
                print(f"No{' Google Street View' if only_google else ''} image found for {lat},{lng}")
                return None, None
            print(f"Found valid Google image at ({real_lat}, {real_lng})")
            await download_image(session, real_lat, real_lng, output_dir)
            return round(float(real_lat), 12), round(float(real_lng), 12)

    return asyncio.run(wrapper())
