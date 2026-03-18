from shapely.ops import unary_union
from shapely.geometry import box
from configs.settings import *
import geopandas as gpd


def main():
    world = gpd.read_file(COUNTRIES_PATH)
    countries = world[(world["CONTINENT"] == "Europe") & (~world["NAME"].isin(EXCLUDE_COUNTRIES))].copy()
    countries["geometry"] = countries["geometry"].intersection(box(*BOUNDARIES))
    countries = countries[~countries.is_empty]

    europe = gpd.GeoDataFrame(geometry=[unary_union(countries["geometry"])], crs=countries.crs)
    europe = europe.explode().to_crs(epsg=3035)
    europe = europe[europe["geometry"].area > MIN_AREA]

    grid_cells = []
    minx, miny, maxx, maxy = europe.total_bounds
    x_offset, y_offset = GRID_OFFSET_X % CELL_SIZE, GRID_OFFSET_Y % CELL_SIZE
    for x in list(range(int(minx - CELL_SIZE + x_offset), int(maxx) + CELL_SIZE, CELL_SIZE)):
        for y in list(range(int(miny - CELL_SIZE + y_offset), int(maxy) + CELL_SIZE, CELL_SIZE)):
            grid_cells.append(box(x, y, x + CELL_SIZE, y + CELL_SIZE))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=europe.crs)
    grid = gpd.overlay(grid, europe, how="intersection")
    grid = grid.explode(index_parts=False).reset_index(drop=True)

    # merge grid parts that are too small
    small_idx = grid[grid.area < MIN_AREA].index
    while len(small_idx) > 0:
        for idx in small_idx:
            geom = grid.loc[idx, "geometry"]
            neighbors = grid[grid.geometry.touches(geom) & (grid.index != idx)].copy()
            if not neighbors.empty:
                neighbors["border"] = neighbors.geometry.apply(lambda n: n.boundary.intersection(geom.boundary).length)
                neighbors = neighbors[neighbors["border"] > 0]
            if not neighbors.empty:
                neighbors["centroid_dist"] = neighbors.geometry.centroid.distance(geom.centroid)
                closest_neighbors = neighbors[neighbors["centroid_dist"] < neighbors["centroid_dist"].min() * 1.1]
                merge_idx = closest_neighbors.geometry.area.idxmin()
                grid.at[merge_idx, "geometry"] = unary_union([grid.at[merge_idx, "geometry"], geom])
                grid = grid.drop(idx)
        small_idx = grid[grid.area < MIN_AREA].index
    grid = grid.reset_index(drop=True)

    europe.to_file(EUROPE_PATH, driver="GPKG")
    grid.to_file(GRID_PATH, driver="GPKG")


if __name__ == "__main__":
    main()
