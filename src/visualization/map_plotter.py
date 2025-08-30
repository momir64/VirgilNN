import matplotlib.pyplot as plt
from configs.settings import *
from shapely import Point
import geopandas as gpd
import json

PROJECTION_TYPE = "EPSG:4326" if False else "EPSG:3035"
ENUMERATE_CELLS = False

with open(ALL_LOCATIONS_PATH) as file:
    grouped_locations = json.load(file)
    points = []
    for locations in grouped_locations.values():
        for location in locations:
            points.append(Point(location["lng"], location["lat"]))

points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
europe = gpd.read_file(EUROPE_PATH).to_crs(PROJECTION_TYPE)
grid = gpd.read_file(GRID_PATH).to_crs(PROJECTION_TYPE)

ax = europe.plot(figsize=(12, 10), color="lightgreen", edgecolor="black")
points_gdf.plot(ax=ax, color="darkgreen", markersize=0.1)
grid.boundary.plot(ax=ax, color="black", linewidth=0.5)

if ENUMERATE_CELLS:
    for idx, row in grid.iterrows():
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        ax.text(x, y, str(idx + 1), ha="center", va="center", fontsize=9, color="green")

ax.set_aspect("auto")
plt.title(f"Europe")
plt.show()
