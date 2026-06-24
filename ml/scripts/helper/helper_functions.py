from scripts.helper.helper_classes import (
    ConfusionMatrix, 
    PerformanceMetrics,
    NumpyDatasetSplit,
    NumpyDataset,
    LoadedTensors,
    TensorDatasetSplit,
    ClassificationEvaluationData,
    Label,
    LSTMConfig,
    OrderBookLSTM,
    OrderBookWindowDataset
)

from scripts.helper.constants import (
    MID_PRICE_COL,
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
    BEST_ASK_COL_IDX,
    BEST_BID_COL_IDX,
    ENGINEERED_FEATURES_PER_ROW,
    TRAIN_TENSOR_PATH,
    VAL_TENSOR_PATH,
    TEST_TENSOR_PATH,
    BEST_ANN_MODEL_PATH,
    ANN_MODEL_CHECKPOINT_PATH,
    BEST_LSTM_MODEL_PATH,
    LSTM_MODEL_CHECKPOINT_PATH,
    device
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# ------------------------------------------
# Preprocessing helpers
# ------------------------------------------
def get_data_from_confusion_matrices(matrices: list[ConfusionMatrix]) -> PerformanceMetrics:
    accuracy_sum = 0
    recall_sum = 0
    precision_sum = 0
    f1_sum = 0

    for matrix in matrices:
        accuracy_sum += matrix.get_accuracy()
        recall_sum += matrix.get_recall()
        precision_sum += matrix.get_precision()
        f1_sum += matrix.get_f1()

    return PerformanceMetrics(
        accuracy = accuracy_sum / len(matrices),
        recall = recall_sum / len(matrices),
        precision = precision_sum / len(matrices),
        f1 = f1_sum / len(matrices)
    )

def scale_dataset(numpy_dataset: NumpyDataset) -> NumpyDataset:
    # Extract np splits from the argument
    train_np_X = numpy_dataset.train_np_split.X
    val_np_X = numpy_dataset.val_np_split.X
    test_np_X = numpy_dataset.test_np_split.X
    
    # Define the preprocessor
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("raw_scaler", StandardScaler())
        ]
    )

    # Push the data through the preprocessor
    train_np_X = preprocessor.fit_transform(train_np_X)
    val_np_X = preprocessor.transform(val_np_X)
    test_np_X = preprocessor.transform(test_np_X)

    # Return the scaled dataset
    train_np_split = NumpyDatasetSplit(X=train_np_X, 
        y=numpy_dataset.train_np_split.y)
    val_np_split = NumpyDatasetSplit(X=val_np_X, 
        y=numpy_dataset.val_np_split.y)
    test_np_split = NumpyDatasetSplit(X=test_np_X, 
        y=numpy_dataset.test_np_split.y)
    return NumpyDataset(
        train_np_split=train_np_split,
        val_np_split=val_np_split,
        test_np_split=test_np_split
    )

def preprocess(numpy_dataset: NumpyDataset) -> NumpyDataset:
    # Extract the X np splits from the argument and add their
    # corresponding features
    train_np_x = add_features(numpy_dataset.train_np_split.X)
    val_np_x = add_features(numpy_dataset.val_np_split.X)
    test_np_x = add_features(numpy_dataset.test_np_split.X)

    # Re-create the numpy dataset
    numpy_dataset.train_np_split.X = train_np_x
    numpy_dataset.val_np_split.X = val_np_x
    numpy_dataset.test_np_split.X = test_np_x

    # Return the transformed numpy dataset back
    return numpy_dataset

def add_features(X_np: np.ndarray):
    X_np = replace_absolute_prices_with_relative_price_distances(
        X_np=X_np)
    X_np = add_spread_pct_feature(X_np=X_np)
    X_np = add_depth_imbalance_features(X_np)
    return X_np

def replace_absolute_prices_with_relative_price_distances(
    X_np: np.ndarray
):
    # Replace the absolute price columns with relative prices
    X_np[:, ASK_PRICE_COLS] = (
        (X_np[:, ASK_PRICE_COLS] - X_np[:, [MID_PRICE_COL]]) /
        X_np[:, [MID_PRICE_COL]]
    )

    X_np[:, BID_PRICE_COLS] = (
        (X_np[:, BID_PRICE_COLS] - X_np[:, [MID_PRICE_COL]]) /
        X_np[:, [MID_PRICE_COL]]
    )

    # Remove the mid price column because now bid/ask prices are already
    # relative to the mid price
    X_np = np.delete(X_np, MID_PRICE_COL, axis=1)

    # Return back the numpy array
    return X_np

def add_spread_pct_feature(X_np: np.ndarray):
    # Extract the best ask and bid columns from the input
    # features
    best_ask_col = X_np[:, BEST_ASK_COL_IDX]
    best_bid_col = X_np[:, BEST_BID_COL_IDX]

    # Compute the spread column
    spread_col = best_ask_col - best_bid_col

    # Append the spread column to the input features
    X_np = np.column_stack((X_np, spread_col))

    # Return the new input features with the spread column
    # added
    return X_np

def add_depth_imbalance_features(X_np: np.ndarray):
    # Find the total ask and bid depth for different depths
    total_ask_depth_1 = 0
    total_ask_depth_5 = 0
    total_ask_depth_10 = 0
    total_bid_depth_1 = 0
    total_bid_depth_5 = 0
    total_bid_depth_10 = 0

    for i, (ask_size_col, bid_size_col) in enumerate(zip(ASK_SIZE_COLS, BID_SIZE_COLS)):
        if i == 0:
            total_bid_depth_1 += X_np[:, bid_size_col]
            total_ask_depth_1 += X_np[:, ask_size_col] 
        elif i >= 1 and i <= 4:
            total_bid_depth_5 += X_np[:, bid_size_col]
            total_ask_depth_5 += X_np[:, ask_size_col]
        else:
            total_bid_depth_10 += X_np[:, bid_size_col]
            total_ask_depth_10 += X_np[:, ask_size_col]

    total_bid_depth_5 += total_bid_depth_1
    total_ask_depth_5 += total_ask_depth_1

    total_ask_depth_10 += total_ask_depth_5
    total_bid_depth_10 += total_bid_depth_5

    # Compute the depth imbalance columns
    depth_imbalance_1 = (
        (total_bid_depth_1 - total_ask_depth_1) /
        (total_bid_depth_1 + total_ask_depth_1)
    ) 
    depth_imbalance_5 = (
        (total_bid_depth_5 - total_ask_depth_5) /
        (total_bid_depth_5 + total_ask_depth_5)
    ) 
    depth_imbalance_10 = (
        (total_bid_depth_10 - total_ask_depth_10) /
        (total_bid_depth_10 + total_ask_depth_10)
    ) 

    # Append the columns to the X_np
    X_np = np.column_stack(
        (X_np, 
         depth_imbalance_1, 
         depth_imbalance_5, 
         depth_imbalance_10)
    )

    # Return the transformed X_np with the depth imbalance
    # columns added
    return X_np

def load_tensors() -> LoadedTensors:
    train_dict = torch.load(TRAIN_TENSOR_PATH)
    val_dict = torch.load(VAL_TENSOR_PATH)
    test_dict = torch.load(TEST_TENSOR_PATH)

    return LoadedTensors(
        train_tensors=TensorDatasetSplit(
            X=train_dict["X"],
            y=train_dict["y"]
        ),
        val_tensors=TensorDatasetSplit(
            X=val_dict["X"],
            y=val_dict["y"]
        ),
        test_tensors=TensorDatasetSplit(
            X=test_dict["X"],
            y=test_dict["y"]
        )
    )

# ------------------------------------------
# Training helpers
# ------------------------------------------
def load_ann_model(resume: bool):
    model = build_ann_model()
    if (resume):
        model_checkpoint_dict = torch.load(
            ANN_MODEL_CHECKPOINT_PATH,
            map_location=device
        )
        model_state_dict = model_checkpoint_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)
    return model

def load_lstm_model(should_resume: bool):
    model = build_lstm_model()
    if should_resume:
        model_checkpoint_dict = torch.load(
            LSTM_MODEL_CHECKPOINT_PATH,
            map_location=device
        )
        model_state_dict = model_checkpoint_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)
    return model

def build_ann_model():
    return nn.Sequential(
        nn.Linear(ENGINEERED_FEATURES_PER_ROW, 32),
        nn.ReLU(),

        nn.Linear(32, 16),
        nn.ReLU(),

        nn.Linear(16, 8),
        nn.ReLU(),

        nn.Linear(8, 3)
    )

def build_lstm_model():
    config = LSTMConfig()
    model = OrderBookLSTM(config)
    return model

def evaluate_majority_baseline(
    train_split: TensorDatasetSplit,
    eval_split: TensorDatasetSplit
):
    # Extract the data from the tensor dataset split
    y_train = train_split.y
    y_test = eval_split.y
    # Figure out the majority class
    up_count = 0
    stationary_count = 0
    down_count = 0
    actual_labels = []
    for label in y_train:
        if label == Label.DOWN: 
            down_count += 1
        elif label == Label.STATIONARY:
            stationary_count += 1
        else:
            up_count += 1

    if (up_count > stationary_count and up_count > down_count):
        majority_label = Label.UP
    elif (down_count > up_count and down_count > stationary_count):
        majority_label = Label.DOWN
    else:
        majority_label = Label.STATIONARY

    # Create list of predictions
    predictions = [majority_label] * len(y_test)
    actual_labels = []
    for label in y_test:
        if label == Label.DOWN:
            actual_labels.append(Label.DOWN)
        elif label == Label.STATIONARY:
            actual_labels.append(Label.STATIONARY)
        else:
            actual_labels.append(Label.UP)

    # Compute respective confusion matrix
    down_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.DOWN
    )
    stationary_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.STATIONARY
    )
    up_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.UP
    )
    down_confusion_matrix = ConfusionMatrix.from_predictions(
        down_eval_data)
    stationary_confusion_matrix = ConfusionMatrix.from_predictions(
        stationary_eval_data)
    up_confusion_matrix = ConfusionMatrix.from_predictions(
        up_eval_data)

    # Calculate performance of the model 
    matrices = [down_confusion_matrix,
        stationary_confusion_matrix, up_confusion_matrix]
    avg_data = get_data_from_confusion_matrices(matrices)
    num_of_correct_predictions = down_eval_data.get_correct_predictions()
    accuracy = num_of_correct_predictions / len(predictions)

    return avg_data.f1

def evaluate_majority_baseline_full_report(
    train_split: TensorDatasetSplit,
    eval_split: TensorDatasetSplit
):
    # Extract the data from the tensor dataset split
    y_train = train_split.y
    y_test = eval_split.y
    # Figure out the majority class
    up_count = 0
    stationary_count = 0
    down_count = 0
    actual_labels = []
    for label in y_train:
        if label == Label.DOWN: 
            down_count += 1
        elif label == Label.STATIONARY:
            stationary_count += 1
        else:
            up_count += 1

    if (up_count > stationary_count and up_count > down_count):
        majority_label = Label.UP
    elif (down_count > up_count and down_count > stationary_count):
        majority_label = Label.DOWN
    else:
        majority_label = Label.STATIONARY

    # Print the stats
    print(f"Down count: {down_count}")
    print(f"Stationary count: {stationary_count}")
    print(f"Up count: {up_count}")

    # Create list of predictions
    predictions = [majority_label] * len(y_test)
    actual_labels = []
    for label in y_test:
        if label == Label.DOWN:
            actual_labels.append(Label.DOWN)
        elif label == Label.STATIONARY:
            actual_labels.append(Label.STATIONARY)
        else:
            actual_labels.append(Label.UP)

    # Compute respective confusion matrix
    down_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.DOWN
    )
    stationary_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.STATIONARY
    )
    up_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.UP
    )
    down_confusion_matrix = ConfusionMatrix.from_predictions(
        down_eval_data)
    stationary_confusion_matrix = ConfusionMatrix.from_predictions(
        stationary_eval_data)
    up_confusion_matrix = ConfusionMatrix.from_predictions(
        up_eval_data)

    # Calculate performance of the model 
    print("Evaluating model performance...")
    matrices = [down_confusion_matrix,
        stationary_confusion_matrix, up_confusion_matrix]
    avg_data = get_data_from_confusion_matrices(matrices)
    num_of_correct_predictions = down_eval_data.get_correct_predictions()
    accuracy = num_of_correct_predictions / len(predictions)

    # Print result
    print("-------------------------")
    print(f"Computation complete.")
    print(f"Total accuracy: {accuracy}")

    print("-------------------------")
    print(f"Down class precision: {down_confusion_matrix.get_precision()}")
    print(f"Down class recall: {down_confusion_matrix.get_recall()}")
    print(f"Down class F1: {down_confusion_matrix.get_f1()}")

    print("-------------------------")
    print(f"Stationary class precision: {stationary_confusion_matrix.get_precision()}")
    print(f"Stationary class recall: {stationary_confusion_matrix.get_recall()}")
    print(f"Stationary class F1: {stationary_confusion_matrix.get_f1()}")

    print("-------------------------")
    print(f"Up class precision: {up_confusion_matrix.get_precision()}")
    print(f"Up class recall: {up_confusion_matrix.get_recall()}")
    print(f"Up class F1: {up_confusion_matrix.get_f1()}")

    print("-------------------------")
    print(f"Average precision: {avg_data.precision}")
    print(f"Average recall: {avg_data.recall}")
    print(f"Average F1: {avg_data.f1}")

def evaluate_ann_model(model: nn.Module, eval_split: TensorDatasetSplit):
    # Print device type
    print(f"Evaluation is using device: {device}")

    # Extract the validation input and labels
    X_val = eval_split.X
    y_val = eval_split.y

    # Set the model to evaluation mode
    model.eval()

    # Batch the validation input before moving to device to reduce
    # chance of device out of memory error
    dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        dataset,
        batch_size=64
    )

    predictions = []
    actual_labels = [Label(value) for value in y_val.tolist()]

    # Compute the model output logits
    with torch.no_grad():
        for X, _ in val_loader:
            # Move X to the device
            X = X.to(device)

            # Forward pass
            logits = model(X)  

            # Convert logits into class predictions
            predictions_batch = torch.argmax(logits, dim=1)

            # Turn the batch's predictions into a list of Labels
            predictions_batch = [
                Label(value) for value in predictions_batch.cpu().tolist()
            ]

            # Append the batch's list of Labels into the overall list
            # of Labels
            predictions.extend(predictions_batch)

    # Compute the model confusion matrix for all three positive classes
    up_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.UP
    )
    stationary_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.STATIONARY
    )
    down_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.DOWN
    )

    down_confusion_matrix = ConfusionMatrix.from_predictions(
        down_eval_data
    )
    stationary_confusion_matrix = ConfusionMatrix.from_predictions(
        stationary_eval_data
    )
    up_confusion_matrix = ConfusionMatrix.from_predictions(
        up_eval_data
    )

    down_f1 = down_confusion_matrix.get_f1()
    stationary_f1 = stationary_confusion_matrix.get_f1()
    up_f1 = up_confusion_matrix.get_f1()
    
    macro_f1 = (down_f1 + stationary_f1 + up_f1) / 3

    return macro_f1

def evaluate_lstm_model(model: nn.Module, val_split: TensorDatasetSplit):
    # Extract the val tensors
    X_val = val_split.X
    y_val = val_split.y

    # Batch the validation input into DataLoader to avoid device
    # out of memory error when moving validation data to device
    val_dataset = OrderBookWindowDataset(
        X_val, 
        y_val, 
        LSTMConfig.sequence_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64
    )

    # Forward pass
    model.eval()
    predictions = []
    actual_labels = []
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(X)

            # Compute predictions
            predictions_batch = torch.argmax(logits, dim=1)
            predictions_batch = [
                Label(value) for value in predictions_batch.tolist()
            ]
            actual_labels_batch = [
                Label(target) for target in y.tolist()
            ]
            predictions.extend(predictions_batch)
            actual_labels.extend(actual_labels_batch)

    # Convert results classification evaluation data
    up_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.UP
    )
    stationary_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.STATIONARY
    )
    down_eval_data = ClassificationEvaluationData(
        predictions=predictions,
        actual_labels=actual_labels,
        positive_class=Label.DOWN
    )

    # Convert classification data into confusion matrix 
    up_confusion_matrix = ConfusionMatrix.from_predictions(
        up_eval_data
    )
    stationary_confusion_matrix = ConfusionMatrix.from_predictions(
        stationary_eval_data
    )
    down_confusion_matrix = ConfusionMatrix.from_predictions(
        down_eval_data
    )

    down_f1 = down_confusion_matrix.get_f1()
    stationary_f1 = stationary_confusion_matrix.get_f1()
    up_f1 = up_confusion_matrix.get_f1()
    
    macro_f1 = (down_f1 + stationary_f1 + up_f1) / 3

    return macro_f1

def update_checkpoint_model(
    model: nn.Module, 
    optimizer: Adam,
    epoch: int,
    macro_f1: float,
    best_macro_f1: float
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "macro_f1": macro_f1,
            "best_macro_f1": best_macro_f1
        },
        ANN_MODEL_CHECKPOINT_PATH
    )

def update_best_model(
        model: nn.Module, 
        optimizer: Adam,
        epoch: int,
        macro_f1: float,
        best_macro_f1: float
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "macro_f1": macro_f1,
            "best_macro_f1": best_macro_f1
        },
        BEST_ANN_MODEL_PATH
    )
    return