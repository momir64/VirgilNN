from configs.settings import API_KEY
from aiolimiter import AsyncLimiter
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import asyncio
import aiohttp
import time

europe = gpd.read_file("data/visualization/europe.gpkg", layer="europe").to_crs("EPSG:4326")
grid = gpd.read_file("data/visualization/grid.gpkg", layer="grid").to_crs("EPSG:4326")
REQUESTS_PER_SECOND = 500
MAX_CONCURRENT_CELLS = 20
LOCATIONS_PER_CELL = 200
RADIUS = 1000

limiter = AsyncLimiter(max_rate=REQUESTS_PER_SECOND, time_period=1)
lock = asyncio.Lock()
completed_cells = 0


async def check_streetview_metadata(session, lat, lon, retries=3):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&radius={RADIUS}&key={API_KEY}"
    for attempt in range(retries):
        try:
            async with limiter:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            status = data.get("status", "UNKNOWN")
            if status == "OK":
                return Point(data["location"]["lng"], data["location"]["lat"])
            elif status == "OVER_QUERY_LIMIT":
                await asyncio.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            await asyncio.sleep(2 ** attempt)
    return None


async def generate_location(session, minx, miny, maxx, maxy):
    while True:
        lat = np.random.uniform(miny, maxy)
        lon = np.random.uniform(minx, maxx)
        point = await check_streetview_metadata(session, lat, lon)
        if point:
            return point


async def generate_locations_for_cell(session, row, start_time, total_cells):
    tasks = [generate_location(session, *row['geometry'].bounds) for _ in range(LOCATIONS_PER_CELL)]
    points = []
    for future in asyncio.as_completed(tasks):
        point = await future
        if row['geometry'].contains(point):
            points.append(point)
        if len(points) >= LOCATIONS_PER_CELL:
            break
    while len(points) < LOCATIONS_PER_CELL:
        point = await generate_location(session, *row['geometry'].bounds)
        if row['geometry'].contains(point):
            points.append(point)

    async with lock:
        global completed_cells
        completed_cells += 1
        elapsed = time.perf_counter() - start_time
        avg_time_per_cell = elapsed / completed_cells
        remaining_cells = total_cells - completed_cells
        eta = remaining_cells * avg_time_per_cell
        print(f"Cell {completed_cells} done. Generated {len(points) * completed_cells} points. "
              f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    return points


async def main():
    points = []
    start_time = time.perf_counter()
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_CELLS * LOCATIONS_PER_CELL)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def process_cell(row):
            async with asyncio.Semaphore(MAX_CONCURRENT_CELLS):
                points.extend(await generate_locations_for_cell(session, row, start_time, len(grid)))

        tasks = [process_cell(row) for idx, row in grid.iterrows()]
        await asyncio.gather(*tasks)

    return points


locations = asyncio.run(main())

# save locations somehow
points_gdf = gpd.GeoDataFrame(geometry=locations, crs=grid.crs)
ax = europe.plot(figsize=(12, 10), color='lightgreen', edgecolor='black')
grid.boundary.plot(ax=ax, color='black', linewidth=0.5)
points_gdf.plot(ax=ax, color='darkgreen', markersize=1)
plt.title(f"Europe Grid Points ({len(locations)} points)")
plt.show()
