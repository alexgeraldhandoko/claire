from pathlib import Path
import torch
import argparse

# File paths
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "train.pt"
RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "onedaybtc.data"
SAVE_FILE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SAVE_FILE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TENSOR_PATH = SAVE_FILE_DIR / "train.pt"
VAL_TENSOR_PATH = SAVE_FILE_DIR / "val.pt"
TEST_TENSOR_PATH = SAVE_FILE_DIR / "test.pt"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
ANN_MODEL_CHECKPOINT_PATH = MODELS_DIR / "ann-model" / "checkpoint.pth"
BEST_ANN_MODEL_PATH = MODELS_DIR / "ann-model" / "best.pth"

# Numerical constants
RANDOM_DATASET_SPLIT_SEED = 42
ORDER_BOOK_LEVELS = 10
NUMBER_OF_LINES_PER_DAY = 10 * 60 * 60 * 24
RAW_FEATURES_PER_ROW = ORDER_BOOK_LEVELS * 2 * 2 + 1
ENGINEERED_FEATURES_PER_ROW = RAW_FEATURES_PER_ROW + 4 - 1
TRAINING_EPOCHS = 20
POLYNOMIAL_TRANSFORMATION_DEGREE = 2

# Column indices
BEST_ASK_COL_IDX = 0
BEST_BID_COL_IDX = 20

ASK_COLS = tuple(range(0, 20))
ASK_PRICE_COLS = tuple(range(0, 20, 2))
ASK_SIZE_COLS = tuple(range(1, 20, 2))

BID_COLS = tuple(range(20, 40))
BID_PRICE_COLS = tuple(range(20, 40, 2))
BID_SIZE_COLS = tuple(range(21, 40, 2))

MID_PRICE_COL = 40
SPREAD_PCT_COL = 41
IMBALANCE_1_COL = 42
IMBALANCE_5_COL = 43
IMBALANCE_10_COL = 44

# Objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_parser = argparse.ArgumentParser(
    description="Parser for training script command-line arguments"
)
train_parser.add_argument("--epochs", type=int, default=20)
train_parser.add_argument("--resume", action="store_true")
preprocessing_parser = argparse.ArgumentParser(
    description="Parser for preprocessing script command-line arguments"
)
preprocessing_parser.add_argument("--shuffle", action="store_true")