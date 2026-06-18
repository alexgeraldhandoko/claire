import sys
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_DIR))

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import AdamW

from scripts.helper.helper_functions import (
    load_tensors,
    load_lstm_model,
    evaluate_lstm_model
)

from scripts.helper.constants import (
    PROCESSED_DATA_DIR,
    LSTM_MODEL_CHECKPOINT_PATH,
    BEST_LSTM_MODEL_PATH,
    device,
    train_parser
)

from scripts.helper.helper_classes import (
    OrderBookWindowDataset,
    LSTMConfig
)

def main():
    print("Starting up training...")
    # Extract command line arguments
    args = train_parser.parse_args()
    epochs = args.epochs
    should_resume = args.resume

    # Load the train, val, and test tensors
    print("Loading tensors...")
    loaded_tensors = load_tensors()
    train_split = loaded_tensors.train_tensors
    val_split = loaded_tensors.val_tensors
    test_split = loaded_tensors.test_tensors

    # Turn the training data into DataLoader
    print("Batching data...")
    config = LSTMConfig()
    X_train = train_split.X
    y_train = train_split.y
    train_dataset = OrderBookWindowDataset(X_train, y_train, config.sequence_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # Load the model
    print("Loading model...")
    model = load_lstm_model(should_resume=should_resume)
    model.to(device)

    # Prepare training variables
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    if should_resume: 
        print("Loading model from previous checkpoint...")
        model_checkpoint_dict = torch.load(
            LSTM_MODEL_CHECKPOINT_PATH,
            map_location=device
        )
        optimizer_state_dict = model_checkpoint_dict[
            "optimizer_state_dict"
        ]
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = model_checkpoint_dict["epoch"] + 1
        end_epoch = start_epoch + epochs
        macro_f1 = model_checkpoint_dict["macro_f1"]
        best_macro_f1 = model_checkpoint_dict["best_macro_f1"]
    else:
        start_epoch = 0
        end_epoch = epochs
        macro_f1 = 0
        best_macro_f1 = float("-inf")

    # Execute training
    for epoch in range(start_epoch, end_epoch):
        model.train()
        # For every batch in the DataLoader
        for X_batch, y_batch in train_loader:
            # Clear weights' gradients from previous batch's computation
            optimizer.zero_grad()
            
            # Move batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute loss
            logits = model(X_batch)
            # logits shape: (64, 3)
            loss = loss_fn(logits, y_batch)

            # Do backpropagation
            loss.backward()
            
            # Do the weight update
            clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )
            optimizer.step()
        
        # Validate the model
        macro_f1 = evaluate_lstm_model(model, val_split)

        # If the model performs better than the best, update the best
        if (macro_f1 > best_macro_f1):
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "macro_f1": macro_f1,
                    "best_macro_f1": best_macro_f1
                },
                BEST_LSTM_MODEL_PATH
            )

        # Save the model into checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "macro_f1": macro_f1,
                "best_macro_f1": best_macro_f1
            },
            LSTM_MODEL_CHECKPOINT_PATH
        )

    # Test the model
    print("Testing the model...")
    best_model_dict = torch.load(
        BEST_LSTM_MODEL_PATH,
        map_location=device
    )
    model_state_dict = best_model_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    test_macro_f1 = evaluate_lstm_model(model, test_split)
    print(f"Test Macro F1: {test_macro_f1}")

if __name__ == "__main__":
    main()
