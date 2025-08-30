from shapely.ops import unary_union
from shapely.geometry import box
from configs.settings import *
import geopandas as gpd

if __name__ == "__main__":
    world = gpd.read_file(COUNTRIES_PATH)
    countries = world[(world['CONTINENT'] == 'Europe') & (~world['NAME'].isin(EXCLUDE_COUNTRIES))].copy()
    countries['geometry'] = countries['geometry'].intersection(box(*BOUNDARIES))
    countries = countries[~countries.is_empty]

    europe = gpd.GeoDataFrame(geometry=[unary_union(countries['geometry'])], crs=countries.crs)
    europe = europe.explode().to_crs(epsg=3035)
    europe = europe[europe['geometry'].area > MIN_AREA]

    grid_cells = []
    minx, miny, maxx, maxy = europe.total_bounds
    for x in list(range(int(minx), int(maxx) + GRID_SIZE, GRID_SIZE)):
        for y in list(range(int(miny), int(maxy) + GRID_SIZE, GRID_SIZE)):
            grid_cells.append(box(x, y, x + GRID_SIZE, y + GRID_SIZE))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=europe.crs)
    grid = gpd.overlay(grid, europe, how='intersection')
    grid = grid.explode(index_parts=False).reset_index(drop=True)

    # merge grid parts that are too small
    small_idx = grid[grid.area < MIN_AREA].index
    while len(small_idx) > 0:
        for idx in small_idx:
            geom = grid.loc[idx, 'geometry']
            neighbors = grid[grid.geometry.touches(geom) & (grid.index != idx)]
            if not neighbors.empty:
                border_lengths = neighbors.geometry.apply(lambda nbr: nbr.boundary.intersection(geom.boundary).length)
                longest_border_idx = border_lengths.idxmax()
                # check if the combined size isn't larger than normal grid cell
                if grid.at[longest_border_idx, 'geometry'].union(geom).area <= GRID_SIZE ** 2:
                    merge_idx = longest_border_idx  # if it's not, use neighbor with the largest shared border for merge
                else:
                    merge_idx = neighbors.geometry.area.idxmin()  # if it is, use the smallest neighbour for merge
                grid.at[merge_idx, 'geometry'] = unary_union([grid.at[merge_idx, 'geometry'], geom])
                grid = grid.drop(idx)
        small_idx = grid[grid.area < MIN_AREA].index
    grid = grid.reset_index(drop=True)

    europe.to_file(EUROPE_PATH, driver="GPKG")
    grid.to_file(GRID_PATH, driver="GPKG")
