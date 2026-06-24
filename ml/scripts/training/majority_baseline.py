import torch

import sys
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_DIR))

from scripts.helper.helper_classes import (
    Label, 
    ConfusionMatrix, 
    ClassificationEvaluationData,
    TensorDatasetSplit
)
from scripts.helper.helper_functions import evaluate_majority_baseline_full_report

from scripts.helper.constants import (
    TRAINING_DATA_PATH,
    TEST_TENSOR_PATH
)

# Load the PyTorch training tensor
print("Loading training data...")
train_dict = torch.load(TRAINING_DATA_PATH)
X_train = train_dict["X"]
y_train = train_dict["y"]

# Load the PyTorch test tensor
test_dict = torch.load(TEST_TENSOR_PATH)
X_test = test_dict["X"]
y_test = test_dict["y"]

# Form the tensor dataset split data types
train_tensor_dataset_split = TensorDatasetSplit(
    X=X_train,
    y=y_train
)
test_tensor_dataset_split = TensorDatasetSplit(
    X=X_test,
    y=y_test
)

# Pass in the tensor data splits into the evaluation function
evaluate_majority_baseline_full_report(train_tensor_dataset_split, 
    test_tensor_dataset_split
)