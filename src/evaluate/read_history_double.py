from configs.settings import HISTORY_PATH
import matplotlib.pyplot as plt
import pickle
import sys


def main(model_name1, epoch_number1, model_name2, epoch_number2):
    with open(HISTORY_PATH(model_name1, epoch_number1), "rb") as f:
        history_dict1 = pickle.load(f)

    with open(HISTORY_PATH(model_name2, epoch_number2), "rb") as f:
        history_dict2 = pickle.load(f)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ['tab:blue', 'tab:orange']

    axs[0].plot(history_dict1['classification_accuracy'], label=f'{model_name1} Train', color=colors[0], linestyle='-')
    axs[0].plot(history_dict1['val_classification_accuracy'], label=f'{model_name1} Validation', color=colors[0], linestyle='--')
    axs[0].plot(history_dict2['classification_accuracy'], label=f'{model_name2} Train', color=colors[1], linestyle='-')
    axs[0].plot(history_dict2['val_classification_accuracy'], label=f'{model_name2} Validation', color=colors[1], linestyle='--')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Classification Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history_dict1['coordinates_haversine'], label=f'{model_name1} Train', color=colors[0], linestyle='-')
    axs[1].plot(history_dict1['val_coordinates_haversine'], label=f'{model_name1} Validation', color=colors[0], linestyle='--')
    axs[1].plot(history_dict2['coordinates_haversine'], label=f'{model_name2} Train', color=colors[1], linestyle='-')
    axs[1].plot(history_dict2['val_coordinates_haversine'], label=f'{model_name2} Validation', color=colors[1], linestyle='--')
    axs[1].set_ylabel('Haversine Distance (km)')
    axs[1].set_title('Coordinate Prediction Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python -m src.evaluate.read_history <model_name1> <epoch_number1> <model_name2> <epoch_number2>")
        sys.exit(1)

    model_name1 = sys.argv[1]
    epoch_number1 = int(sys.argv[2])
    model_name2 = sys.argv[3]
    epoch_number2 = int(sys.argv[4])
    main(model_name1, epoch_number1, model_name2, epoch_number2)
