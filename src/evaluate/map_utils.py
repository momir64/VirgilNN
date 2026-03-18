from shapely.geometry import Point
import matplotlib.pyplot as plt
from configs.settings import *
import geopandas as gpd
import numpy as np


def get_grid_cell_centers():
    grid = gpd.read_file(GRID_PATH)
    grid_metric = grid.to_crs(grid.estimate_utm_crs())
    centroids = grid_metric.geometry.centroid.to_crs("EPSG:4326")
    return np.array([[point.y, point.x] for point in centroids], dtype=np.float32)


def get_map_data(lat, lon):
    europe = gpd.read_file(EUROPE_PATH).to_crs(PROJECTION_TYPE)
    grid = gpd.read_file(GRID_PATH).to_crs(PROJECTION_TYPE)
    point = None
    try:
        point = Point(float(lon), float(lat))
    except (TypeError, ValueError):
        pass
    return europe, grid, point


def draw_map(ax, lat, lon, show_grid=True, predicted_gps=None, softmax=None):
    ax.clear()
    ax.set_facecolor("lightskyblue")

    europe, grid, point = get_map_data(lat, lon)
    europe.plot(ax=ax, color="lightgreen", edgecolor="black")

    if show_grid:
        if softmax is not None:
            grid["intensity"] = [softmax.get(idx, 0) for idx in grid.index]
            grid.plot(ax=ax, column="intensity", cmap="Greens", edgecolor="black", linewidth=0.5)
        else:
            grid.boundary.plot(ax=ax, color="black", linewidth=0.5)

    if point:
        gdf_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs(PROJECTION_TYPE)
        gdf_point.plot(ax=ax, color="red", marker='x', markersize=50)

    if predicted_gps is not None:
        pred_lat, pred_lon = predicted_gps
        pred_point = gpd.GeoDataFrame(geometry=[Point(pred_lon, pred_lat)], crs="EPSG:4326").to_crs(PROJECTION_TYPE)
        pred_point.plot(ax=ax, color="darkorange", marker='o', markersize=30)

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
