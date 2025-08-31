# Vision-based Image Recognition Geoencoder for Identifying Locations using a Neural Network

## Project setup
```shell
pip install -r requirements.txt
```

## Dataset gathering
```shell
python -m src.data.map_generator         # generating grid map
python -m src.data.locations_generator   # generating random valid locations
python -m src.data.locations_splitter    # splitting locations into smaller sets per cell
python -m src.data.batch_generator       # generating request batches for downloading
python -m src.data.batch_downloader      # downloading single batch
```

## Visualization
```shell
python -m src.visualization.map_plotter
```