import matplotlib.pyplot as plt
from configs.settings import *
from shapely import Point
import geopandas as gpd
import json

if __name__ == "__main__":
    europe = gpd.read_file(EUROPE_PATH).to_crs(PROJECTION_TYPE)
    grid = gpd.read_file(GRID_PATH).to_crs(PROJECTION_TYPE)

    ax = europe.plot(figsize=FIG_SIZE, color="lightgreen", edgecolor="black")
    grid.boundary.plot(ax=ax, color="black", linewidth=0.5)

    if LOAD_LOCATIONS:
        with open(PLOT_LOCATIONS_PATH) as file:
            grouped_locations = json.load(file)
            points = []
            for locations in grouped_locations.values():
                for location in locations:
                    points.append(Point(location["lng"], location["lat"]))
        points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(PROJECTION_TYPE)
        points_gdf.plot(ax=ax, color="darkgreen", markersize=0.1)

    if ENUMERATE_CELLS:
        for idx, row in grid.iterrows():
            x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
            ax.text(x, y, str(idx + 1), ha="center", va="center", fontsize=9, color="green")

    ax.set_aspect("auto")
    plt.title(f"Europe")
    plt.show()
