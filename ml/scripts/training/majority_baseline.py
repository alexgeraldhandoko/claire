import torch
from pathlib import Path

from ml.scripts.helper.helper_classes import Label, ConfusionMatrix, ClassificationEvaluationData
from ml.scripts.helper.helper_functions import get_data_from_confusion_matrices

from ml.scripts.helper.constants import TRAINING_DATA_PATH

# Load the PyTorch training tensor
print("Loading training data...")
train_dict = torch.load(TRAINING_DATA_PATH)
X_train = train_dict["X_train"]
y_train = train_dict["y_train"]

# Figure out the majority class
up_count = 0
stationary_count = 0
down_count = 0
actual_labels = []
for label in y_train:
    if label == Label.DOWN: 
        down_count += 1
        actual_labels.append(Label.DOWN)
    elif label == Label.STATIONARY:
        stationary_count += 1
        actual_labels.append(Label.STATIONARY)
    else:
        up_count += 1
        actual_labels.append(Label.UP)

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
print("Computing predictions...")
predictions = [majority_label] * len(y_train)

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