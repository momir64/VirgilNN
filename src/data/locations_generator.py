from src.data.utils import save_locations
from aiolimiter import AsyncLimiter
from shapely.geometry import Point
from configs.settings import *
import geopandas as gpd
import numpy as np
import asyncio
import aiohttp
import time

limiter = AsyncLimiter(max_rate=REQUESTS_PER_SECOND, time_period=1)
cell_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CELLS)
grid = gpd.read_file(GRID_PATH).to_crs("EPSG:4326")
lock = asyncio.Lock()
completed_cells = 0


async def check_streetview_metadata(session, lat, lng, retries=3):
    url = f"{STREETVIEW_API_URL}/metadata?key={API_KEY}&radius={RADIUS}&location={lat},{lng}"
    for attempt in range(retries):
        try:
            async with limiter:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            status = data.get("status", "UNKNOWN")
            if status == "OK" and "location" in data and "lng" in data["location"] and "lat" in data["location"]:
                if not USE_ONLY_MADE_BY_GOOGLE or ("copyright" in data and data["copyright"] == "© Google"):
                    return Point(data["location"]["lng"], data["location"]["lat"])
                else:
                    return None
            elif status == "OVER_QUERY_LIMIT":
                await asyncio.sleep(2 ** attempt)
            else:
                return None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            await asyncio.sleep(2 ** attempt)
    return None


async def generate_location(session, geom):
    minx, miny, maxx, maxy = geom.bounds
    while True:
        lat = np.random.uniform(miny, maxy)
        lng = np.random.uniform(minx, maxx)
        if geom.contains(Point(lng, lat)):
            point = await check_streetview_metadata(session, lat, lng)
            if point and geom.contains(point):
                return point


async def generate_locations_for_cell(session, cell, start_time, total_cells):
    locations = set()
    while len(locations) < LOCATIONS_PER_CELL:
        tasks = [generate_location(session, cell["geometry"]) for _ in range(LOCATIONS_PER_CELL - len(locations))]
        results = await asyncio.gather(*tasks)
        locations.update((point.x, point.y) for point in results)

    # Logging
    async with lock:
        global completed_cells
        completed_cells += 1
        elapsed = time.perf_counter() - start_time
        eta = (total_cells - completed_cells) * (elapsed / completed_cells)
        print(f"Cell {completed_cells} done. Generated {len(locations) * completed_cells} locations. "
              f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    return [{"lat": lat, "lng": lng} for lng, lat in locations]


async def process_cell(session, cell, start_time, total_cells, locations, cell_id):
    async with cell_semaphore:
        cell_locations = await generate_locations_for_cell(session, cell, start_time, total_cells)
    async with lock:
        locations[cell_id] = cell_locations


async def generate_locations():
    locations = {}
    start_time = time.perf_counter()
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_CELLS * LOCATIONS_PER_CELL)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_cell(session, cell, start_time, len(grid), locations, idx) for idx, cell in grid.iterrows()]
        await asyncio.gather(*tasks)

    return locations


if __name__ == "__main__":
    locations_per_cell = asyncio.run(generate_locations())
    save_locations(locations_per_cell, ALL_LOCATIONS_PATH)
    print(f"Locations saved in {ALL_LOCATIONS_PATH}")
