from configs.settings import API_KEY
from aiolimiter import AsyncLimiter
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import asyncio
import aiohttp
import time

europe = gpd.read_file("data/visualization/europe.gpkg").to_crs("EPSG:4326")
grid = gpd.read_file("data/visualization/grid.gpkg").to_crs("EPSG:4326")
REQUESTS_PER_SECOND = 500
MAX_CONCURRENT_CELLS = 5
LOCATIONS_PER_CELL = 200
RADIUS = 1000  # 1 km

limiter = AsyncLimiter(max_rate=REQUESTS_PER_SECOND, time_period=1)
cell_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CELLS)
lock = asyncio.Lock()
completed_cells = 0


async def check_streetview_metadata(session, lat, lon, retries=3):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?key={API_KEY}&radius={RADIUS}&location={lat},{lon}"
    for attempt in range(retries):
        try:
            async with limiter:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            status = data.get("status", "UNKNOWN")
            if status == "OK" and "location" in data and "lng" in data["location"] and "lat" in data["location"]:
                return Point(data["location"]["lng"], data["location"]["lat"])
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
        lon = np.random.uniform(minx, maxx)
        if geom.contains(Point(lon, lat)):
            point = await check_streetview_metadata(session, lat, lon)
            if point and geom.contains(point):
                return point


async def generate_locations_for_cell(session, cell, start_time, total_cells):
    tasks = [generate_location(session, cell['geometry']) for _ in range(LOCATIONS_PER_CELL)]
    points = await asyncio.gather(*tasks)

    # Logging
    async with lock:
        global completed_cells
        completed_cells += 1
        elapsed = time.perf_counter() - start_time
        eta = (total_cells - completed_cells) * (elapsed / completed_cells)
        print(f"Cell {completed_cells} done. Generated {len(points) * completed_cells} points. "
              f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    return points


async def process_cell(cell, session, points, start_time, total_cells):
    async with cell_semaphore:
        cell_points = await generate_locations_for_cell(session, cell, start_time, total_cells)
    async with lock:
        points.extend(cell_points)


async def generate_locations():
    points = []
    start_time = time.perf_counter()
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_CELLS * LOCATIONS_PER_CELL)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_cell(cell, session, points, start_time, len(grid)) for idx, cell in grid.iterrows()]
        await asyncio.gather(*tasks)

    return points


locations = asyncio.run(generate_locations())

# todo: save locations somehow instead of displaying them
points_gdf = gpd.GeoDataFrame(geometry=locations, crs=grid.crs)
ax = europe.plot(figsize=(12, 10), color='lightgreen', edgecolor='black')
grid.boundary.plot(ax=ax, color='black', linewidth=0.5)
points_gdf.plot(ax=ax, color='darkgreen', markersize=1)
plt.title(f"Europe Grid Points ({len(locations)} points)")
plt.show()
