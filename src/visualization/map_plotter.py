import matplotlib.pyplot as plt
import geopandas as gpd

europe = gpd.read_file("data/intermediate/europe.gpkg", layer="europe")
grid = gpd.read_file("data/intermediate/grid.gpkg", layer="grid")

ax = europe.plot(figsize=(12, 10), color='lightgreen', edgecolor='black')
grid.boundary.plot(ax=ax, color='black', linewidth=0.5)
for idx, row in grid.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, str(idx + 1), ha='center', va='center', fontsize=9, color='green')

plt.title(f"Europe")
plt.show()
