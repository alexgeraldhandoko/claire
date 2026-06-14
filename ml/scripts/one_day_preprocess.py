# ------------------------------------------
# Imports
# ------------------------------------------
# ML libraries imports
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Import Python algorithm/data structures helper libraries
from sortedcontainers import SortedDict

# File-handling imports
from pathlib import Path
import orjson

# ------------------------------------------
# Constants
# ------------------------------------------
print("Initialising constants")
# Prepare the TreeMap of the whole order book
# This keeps track of the ordering and size of the bids and asks
# The key will be the order amount - this is used to query the top 10
# The value will be the order size
bid_order_book_as_map = SortedDict()
ask_order_book_as_map = SortedDict()

# Load the one day file
print("Loading files from memory...")
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "onedaybtc.data"
SAVE_FILE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SAVE_FILE_DIR.mkdir(parents=True, exist_ok=True)

# Other constants
RANDOM_DATASET_SPLIT_SEED = 42
ORDER_BOOK_LEVELS = 10
NUMBER_OF_LINES_PER_DAY = 10 * 60 * 60 * 24
FEATURES_PER_ROW = ORDER_BOOK_LEVELS * 2 * 2 + 1

# Prepare the np arrays to record all the order books 
# for every 0.1s for the top 10 levels of bid and ask
# Shape: [860000, 41]
# 860,000 rows of [ask_price_1, ask_size_1, ask_price_2, ask_size_2, ...]
X_np = np.empty((NUMBER_OF_LINES_PER_DAY, FEATURES_PER_ROW), dtype=np.float32)

# ------------------------------------------
# Main preprocessing
# ------------------------------------------
# For each line, 
# 1. Process it to change the TreeMap
# 2. Construct it into an order book
# 3. Push the order book as a new row in the PyTorch tensor
with RAW_DATA_PATH.open("rb") as file:
    print("Starting main execution...")
    curr_valid_row = 0
    for line_number, line in enumerate(file):
        if (line_number % 10000 == 0):
            print(f"Processed {line_number + 1} lines")

        # Parse the line JSON into an object
        json_line = orjson.loads(line)

        bids = json_line["data"]["b"]
        asks = json_line["data"]["a"]

        # Overwrite the sorted dict if the line is a snapshot
        if json_line["type"] == "snapshot":
            bid_order_book_as_map.clear()
            ask_order_book_as_map.clear()
            for bid in bids:
                bid_amount = float(bid[0])
                bid_share_size = float(bid[1])
                bid_order_book_as_map[bid_amount] = bid_share_size
            for ask in asks:
                ask_amount = float(ask[0])
                ask_share_size = float(ask[1])
                ask_order_book_as_map[ask_amount] = ask_share_size

        # Update the sorted dict if the line is a delta
        else:
            for bid in bids:
                bid_amount = float(bid[0])
                bid_share_size = float(bid[1])
                if bid_share_size == 0:
                    bid_order_book_as_map.pop(bid_amount, None)
                else:
                    bid_order_book_as_map[bid_amount] = bid_share_size
            for ask in asks:
                ask_amount = float(ask[0])
                ask_share_size = float(ask[1])
                if ask_share_size == 0:
                    ask_order_book_as_map.pop(ask_amount, None)
                else:
                    ask_order_book_as_map[ask_amount] = ask_share_size

        # Search for the top 10 bids and top 10 asks
        if (
            len(ask_order_book_as_map) < 10 or
            len(bid_order_book_as_map) < 10
        ):
            continue

        col_index = 0
        for i in range(ORDER_BOOK_LEVELS):
            ask_price, ask_shares = ask_order_book_as_map.peekitem(i)
            X_np[curr_valid_row, col_index] = ask_price
            X_np[curr_valid_row, col_index + 1] = ask_shares
            col_index += 2
        for i in range(ORDER_BOOK_LEVELS):
            bid_price, bid_shares = bid_order_book_as_map.peekitem(-(i + 1))
            X_np[curr_valid_row, col_index] = bid_price
            X_np[curr_valid_row, col_index + 1] = bid_shares
            col_index += 2
        mid_price = (ask_order_book_as_map.peekitem(0)[0]
            + bid_order_book_as_map.peekitem(-1)[0]) / 2
        X_np[curr_valid_row, col_index] = mid_price
        curr_valid_row += 1

# Remove any skipped lines above
X_np = X_np[:curr_valid_row]

# Create the ground truth classification levels from the 
# mid price movement from one time step to the next time step
print("Creating ground truth labels...")

best_ask_col_idx = 0
best_bid_col_idx = ORDER_BOOK_LEVELS * 2
best_ask_np = X_np[:, best_ask_col_idx]
best_bid_np = X_np[:, best_bid_col_idx]
mid_price_np = (best_ask_np + best_bid_np) / 2
mid_price_change_np = (
    (mid_price_np[1:] - mid_price_np[:-1]) / mid_price_np[:-1]
)
spread_pct = np.median(
    (best_ask_np[:-1] - best_bid_np[:-1]) / mid_price_np[:-1]
)

classification_np = np.ones(len(mid_price_change_np), dtype=np.int64)
classification_np[mid_price_change_np > spread_pct] = 2
classification_np[mid_price_change_np < -spread_pct] = 0

X_np = X_np[:-1]

# Split the data into training, validation, and test sets
print("Splitting data into train, val, and test sets...")
X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
    X_np,
    classification_np,
    random_state=RANDOM_DATASET_SPLIT_SEED,
    train_size=0.7
)
X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(
    X_temp_np,
    y_temp_np,
    random_state=RANDOM_DATASET_SPLIT_SEED,
    train_size=0.5,
)

# Convert the dataframes into PyTorch tensors
print("Converting data into PyTorch tensors...")
X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np)
X_val_tensor = torch.from_numpy(X_val_np)
y_val_tensor = torch.from_numpy(y_val_np)
X_test_tensor = torch.from_numpy(X_test_np)
y_test_tensor = torch.from_numpy(y_test_np)

# Save the resulting PyTorch tensor that represents the order books
# for one day
print("Saving PyTorch tensors...")
torch.save(
    {
        "X_train": X_train_tensor,
        "y_train": y_train_tensor
    },
    SAVE_FILE_DIR / "train.pt"
)
torch.save(
    {
        "X_val": X_val_tensor,
        "y_val": y_val_tensor
    },
    SAVE_FILE_DIR / "val.pt"
)
torch.save(
    {
        "X_test": X_test_tensor,
        "y_test": y_test_tensor
    },
    SAVE_FILE_DIR / "test.pt"
)
print("Preprocessing complete.")