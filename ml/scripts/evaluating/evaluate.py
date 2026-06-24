import torch

from pathlib import Path
import sys
ML_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_DIR))

from scripts.helper.helper_classes import (
    TensorDatasetSplit
)

from scripts.helper.constants import (
    BEST_ANN_MODEL_PATH,
    BEST_LSTM_MODEL_PATH,
    TEST_TENSOR_PATH,
    TRAIN_TENSOR_PATH,
    device
)

from scripts.helper.helper_functions import (
    build_ann_model,
    build_lstm_model,
    evaluate_ann_model,
    evaluate_lstm_model,
    evaluate_majority_baseline
)

def main():
    # Load the training data
    train_data = torch.load(TRAIN_TENSOR_PATH,
        map_location=device
    )
    X_train = train_data["X"]
    y_train = train_data["y"]
    train_tensor_dataset_split = TensorDatasetSplit(
        X=X_train,
        y=y_train
    )

    # Load the testing data
    test_data = torch.load(TEST_TENSOR_PATH,
        map_location=device
    )
    X_test = test_data["X"]
    y_test = test_data["y"]
    test_tensor_dataset_split = TensorDatasetSplit(
        X=X_test,
        y=y_test
    )

    # Load the different models
    ann_model = build_ann_model()
    lstm_model = build_lstm_model()
    ann_model_dict = torch.load(BEST_ANN_MODEL_PATH,
        map_location=device
    )
    lstm_model_dict = torch.load(BEST_LSTM_MODEL_PATH,
        map_location=device
    )
    ann_model.load_state_dict(ann_model_dict["model_state_dict"])
    lstm_model.load_state_dict(lstm_model_dict["model_state_dict"])

    # Test the different models on testing data
    majority_baseline_macro_f1 = evaluate_majority_baseline(
        train_tensor_dataset_split, 
        test_tensor_dataset_split
    )   
    ann_macro_f1 = evaluate_ann_model(
        ann_model,
        test_tensor_dataset_split
    )
    lstm_macro_f1 = evaluate_lstm_model(
        lstm_model,
        test_tensor_dataset_split
    )

    # Print the test results
    print(f"Majority Baseline Macro F1: {majority_baseline_macro_f1}")
    print(f"ANN Macro F1: {ann_macro_f1}")
    print(f"LSTM Macro F1: {lstm_macro_f1}")

if __name__ == "__main__":
    main()