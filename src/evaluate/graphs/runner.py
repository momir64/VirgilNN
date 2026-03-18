from src.evaluate.graphs import reg_weighted_distance_error_distribution
from src.evaluate.graphs import reg_top_distance_error_distribution
from src.evaluate.graphs import softmax_entropy_distribution
from src.evaluate.graphs import top_softmax_distribution
from src.evaluate.graphs import distance_distribution
from src.evaluate.graphs import map_distance_error
from src.evaluate.graphs import plot_class_balance
from src.evaluate.graphs import map_class_accuracy
from src.evaluate.graphs import confusion_matrix
from src.evaluate.graphs import roc_curve
from .utils import load_test_data
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluate.graphs.runner <model_name1> [<model_name2> ...]")
        sys.exit(1)

    for model_name in sys.argv[1:]:
        print(f"\tLoading test data and model for {model_name}...")
        test_ds, model = load_test_data(model_name)

        print(f"\tCreating class balance graph for {model_name}")
        plot_class_balance.main("middle" in model_name)

        print(f"\tCreating distance distribution graph for {model_name}...")
        distance_distribution.main(model_name, test_ds, model)

        print(f"\tCreating distance error map for {model_name}...")
        map_distance_error.main(model_name, test_ds, model)

        print(f"\tCreating class accuracy map for {model_name}...")
        map_class_accuracy.main(model_name, test_ds, model)

        print(f"\tCreating confusion matrix for {model_name}...")
        confusion_matrix.main(model_name, test_ds, model)

        print(f"\tCreating ROC curve graph for {model_name}...")
        roc_curve.main(model_name, test_ds, model)

        print(f"\tSoftmax entropy distribution for {model_name}...")
        softmax_entropy_distribution.main(model_name, test_ds, model)

        print(f"\tTop softmax distribution for {model_name}...")
        top_softmax_distribution.main(model_name, test_ds, model)

        print(f"\tCreating regression vs weighted classification distance error distribution for {model_name}...")
        reg_weighted_distance_error_distribution.main(model_name, test_ds, model)

        print(f"\tCreating regression vs top classification distance error distribution for {model_name}...")
        reg_top_distance_error_distribution.main(model_name, test_ds, model)
