from configs.settings import HISTORY_PATH
import matplotlib.pyplot as plt
import pickle
import sys
import os


def main(model_name, end_epoch):
    history_dict = {
        'classification_accuracy': [],
        'val_classification_accuracy': [],
        'coordinates_haversine': [],
        'val_coordinates_haversine': []
    }

    for epoch in range(end_epoch + 1):
        path = HISTORY_PATH(model_name, epoch)
        if not os.path.exists(path):
            print(f"Warning: File not found for epoch {epoch}: {path}")
            continue

        with open(path, "rb") as f:
            epoch_data = pickle.load(f)

        for key in history_dict:
            value = epoch_data.get(key)
            if value is None:
                print(f"Warning: '{key}' not found in epoch {epoch}")
                history_dict[key].append(None)
            else:
                history_dict[key].append(value)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(history_dict['classification_accuracy'], label='Train Classification')
    axs[0].plot(history_dict['val_classification_accuracy'], label='Val Classification')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Classification Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history_dict['coordinates_haversine'], label='Train Haversine')
    axs[1].plot(history_dict['val_coordinates_haversine'], label='Val Haversine')
    axs[1].set_ylabel('Haversine Distance (km)')
    axs[1].set_title('Coordinate Prediction Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.evaluate.read_history <model_name> <epoch_number>")
        sys.exit(1)

    model_name = sys.argv[1]
    epoch_number = int(sys.argv[2])
    main(model_name, epoch_number)
