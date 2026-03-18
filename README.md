# Vision-based Image Recognition Geocoder for Identifying Locations using a Neural Network

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

## Dataset visualization
```shell
python -m src.visualization.map_plotter
```

## Dataset preprocessing
```shell
python -m src.preprocess.dataset_scaler        # scaling images by factor of 2
python -m src.preprocess.panorama_generator    # stiching images into panoramas
python -m src.preprocess.dataset_agumentation  # horizontal flipping images
```

## Model training
```shell
python -m src.model.early_integration.train_early <model_name>
python -m src.model.middle_integration.train_middle_single <model_name>
python -m src.model.middle_integration.train_middle_full  <step1_model_name> <model_name>
```

## Evaluation
```shell
python -m src.evaluate.gui [<lattitude>,<longitude>]         # GUI tool that uses pretrained models
python -m src.graphs.runner <model_name> [<model_name> ...]  # creates all graphs for specified models
```