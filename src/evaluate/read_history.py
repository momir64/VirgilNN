from configs.settings import HISTORY_PATH
import matplotlib.pyplot as plt
import pickle
import sys


def main(model_name, epoch):
    with open(HISTORY_PATH(model_name, epoch), "rb") as f:
        history_dict = pickle.load(f)

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
