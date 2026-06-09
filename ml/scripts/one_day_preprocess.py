# ------------------------------------------
# Imports
# ------------------------------------------
# ML libraries imports
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Import Python algorithm/data structures helper libraries
from sortedcontainers import SortedDict

# File-handling imports
from pathlib import Path
import json

# ------------------------------------------
# Constants
# ------------------------------------------
# Prepare the PyTorch tensor to record all the order books 
# for every 0.1s for the top 10 levels of bid and ask
# Shape: [860000, 41]
# 860,000 rows of [bid_price_1, bid_size_1, bid_price_2, bid_size_2, ...]
X_df = pd.DataFrame(columns=range(41))
mid_price_df = pd.Series(dtype="float64")
classification_df = pd.Series(dtype="int64")

# Prepare the TreeMap of the whole order book
# This keeps track of the ordering and size of the bids and asks
# The key will be the order amount - this is used to query the top 10
# The value will be the order size
bid_order_book_as_map = SortedDict()
ask_order_book_as_map = SortedDict()

# Load the one day file
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "onedaybtc.data"
SAVE_FILE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SAVE_FILE_DIR.mkdir(parents=True, exist_ok=True)

# Other constants
RANDOM_DATASET_SPLIT_SEED = 42

# ------------------------------------------
# Main preprocessing
# ------------------------------------------
# For each line, 
# 1. Process it to change the TreeMap
# 2. Construct it into an order book
# 3. Push the order book as a new row in the PyTorch tensor
with RAW_DATA_PATH.open("r") as file:
    previous_row = None
    for line_number, line in enumerate(file):
        # Prepare the list that represents the new row to be
        # added to the PyTorch tensor
        new_row = []

        # Parse the line JSON into an object
        stripped_line = line.strip()
        json_line = json.loads(stripped_line)

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
        for count, price in enumerate(reversed(bid_order_book_as_map)):
            shares = bid_order_book_as_map[price]
            new_row.append(price)
            new_row.append(shares)
            if count == 9:
                break

        for count, price in enumerate(ask_order_book_as_map):
            shares = ask_order_book_as_map[price]
            new_row.append(price)
            new_row.append(shares)
            if count == 9:
                break
        
        # The new row needs at least 10 top bid and ask levels
        # respectively, and if the current order book doesn't 
        # have that many, then skip the current order book
        if len(new_row) != 40:
            continue

        # Calculate the mid price
        best_bid = next(reversed(bid_order_book_as_map))
        best_ask = next(iter(ask_order_book_as_map))
        mid_price = (best_bid + best_ask) / 2
        new_row.append(mid_price)

        # Put the new row list as a new row in the X and y dataframes
        # Exclude the very last line because we don't have a label
        # classification for it (we don't know the next time step's
        # label classification after the last time step)
        if previous_row is not None:
            X_df.loc[len(X_df)] = previous_row
        mid_price_df.loc[len(mid_price_df)] = mid_price

        # To keep track of whether we are in the first 
        previous_row = new_row

# Create the ground truth classification levels from the 
# mid price movement from one time step to the next time step
for i in range(len(mid_price_df) - 1):
    curr_mid_price = mid_price_df[i]
    next_mid_price = mid_price_df[i + 1]
    mid_price_change = next_mid_price - curr_mid_price
    if abs(mid_price_change) < 0.001:
        classification_df[i] = 1
    elif mid_price_change > 0.001:
        classification_df[i] = 2
    else:
        classification_df[i] = 0

# Split the data into training, validation, and test sets
X_train_df, X_temp_df, y_train_df, y_temp_df = train_test_split(
    X_df,
    classification_df,
    shuffle=True,
    stratify=classification_df,
    random_state=RANDOM_DATASET_SPLIT_SEED,
    train_size=0.7
)
X_val_df, X_test_df, y_val_df, y_test_df = train_test_split(
    X_temp_df,
    y_temp_df,
    random_state=RANDOM_DATASET_SPLIT_SEED,
    train_size=0.5,
    shuffle=True,
    stratify=y_temp_df
)

# Convert the dataframes into PyTorch tensors
X_train_tensor = torch.tensor(X_train_df.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_df.to_numpy(), dtype=torch.long)
X_val_tensor = torch.tensor(X_val_df.to_numpy(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_df.to_numpy(), dtype=torch.long)
X_test_tensor = torch.tensor(X_test_df.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_df.to_numpy(), dtype=torch.long)

# Save the resulting PyTorch tensor that represents the order books
# for one day
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