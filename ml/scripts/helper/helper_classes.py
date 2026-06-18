from enum import IntEnum
from dataclasses import dataclass

from scripts.helper.math_utils import divide_or_zero

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

class Label(IntEnum):
    DOWN = 0
    STATIONARY = 1
    UP = 2

class ClassificationEvaluationData:
    predictions: list[Label]
    actual_labels: list[Label]
    positive_class: Label

    def __init__(self, 
        predictions: list[Label], 
        actual_labels: list[Label], 
        positive_class: Label
    ):
        self.predictions = predictions
        self.actual_labels = actual_labels
        self.positive_class = positive_class
    
    def get_correct_predictions(self):
        correct_predictions = 0
        for prediction, label in zip(self.predictions, self.actual_labels):
            if prediction == label:
                correct_predictions += 1
        return correct_predictions

@dataclass
class PerformanceMetrics:
    accuracy: float
    f1: float
    recall: float
    precision: float

class ConfusionMatrix:
    def __init__(self,
        true_positive: int,
        true_negative: int,
        false_negative: int,
        false_positive: int
    ):
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_negative = false_negative
        self.false_positive = false_positive

    @classmethod
    def from_predictions(
        cls,
        classification_evaluation_data: ClassificationEvaluationData
    ):
        # Extract information from the arguments
        predictions = classification_evaluation_data.predictions
        positive_class = classification_evaluation_data.positive_class
        actual_labels = classification_evaluation_data.actual_labels

        # Calculate the confusion matrix
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for prediction, label in zip(predictions, actual_labels):
            if prediction == positive_class:
                if prediction == label:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if label == positive_class:
                    false_negative += 1
                else:
                    true_negative += 1
        
        # Create and return the confusion matrix
        return cls(
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative
        )
    
    def get_num_of_correct_predictions(self):
        return self.true_positive + self.true_negative

    def get_accuracy(self) -> float:
        correct_predictions = self.true_positive + self.true_negative
        total_predictions = (self.true_positive + self.true_negative
            + self.false_negative + self.false_positive)
        return correct_predictions / total_predictions

    def get_precision(self) -> float:
        return divide_or_zero(self.true_positive, 
            (self.true_positive + self.false_positive))

    def get_recall(self) -> float:
        return divide_or_zero(self.true_positive,
            (self.true_positive + self.false_negative))

    def get_f1(self) -> float:
        precision = self.get_precision()
        recall = self.get_recall()

        return divide_or_zero(2 * precision * recall,
            precision + recall)
    
@dataclass
class NumpyDatasetSplit:
    X: np.array
    y: np.array

@dataclass
class NumpyDataset:
    train_np_split: NumpyDatasetSplit
    val_np_split: NumpyDatasetSplit
    test_np_split: NumpyDatasetSplit

@dataclass
class TensorDatasetSplit:
    X: torch.Tensor
    y: torch.Tensor

@dataclass
class LoadedTensors:
    train_tensors: TensorDatasetSplit
    val_tensors: TensorDatasetSplit
    test_tensors: TensorDatasetSplit

@dataclass(frozen=True)
class LSTMConfig:
    input_size: int = 44
    hidden_size: int = 64
    num_layers: int = 2
    num_classes: int = 3
    sequence_length: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    is_bidirectional: bool = False

class OrderBookLSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super.__init__()

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.is_bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
    
    def forward(self, X: torch.Tensor):
        _, (h_n, _) = self.lstm(X)
        # h_n has shape (2, 64, 64)

        logits = self.classifier(h_n[1, :, :])
        # logits has shape (64, 3)
        return logits
    
class OrderBookWindowDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, sequence_length: int):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx: int):
        start = min(idx, len(self.X) - 1)
        end = min(idx + self.sequence_length, len(self.X) - 1)
        X_window = self.X[start:end]
        y_target = self.y[max(end - 1, 0)]
        return X_window, y_target

















