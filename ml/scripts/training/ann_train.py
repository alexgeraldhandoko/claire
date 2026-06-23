from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_DIR))

from scripts.helper.constants import (
    ANN_MODEL_CHECKPOINT_PATH,
    BEST_ANN_MODEL_PATH,
    device,
    train_parser
)
from scripts.helper.helper_classes import TensorDatasetSplit, LoadedTensors
from scripts.helper.helper_functions import (
    load_tensors, 
    evaluate_ann_model,
    update_best_model,
    update_checkpoint_model,
    load_ann_model
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

def main():
    # Print device type
    print(f"The following device is available: {device}")

    # Load the command-line arguments
    args = train_parser.parse_args()

    # Load the training, val, and test tensors
    print("Loading the tensor data")
    loaded_tensors = load_tensors()

    train_split = loaded_tensors.train_tensors
    val_split = loaded_tensors.val_tensors
    test_split = loaded_tensors.test_tensors

    X_train = train_split.X
    y_train = train_split.y

    # Use a loader to pass in batches to the model
    # during training
    print("Processing data into batches...")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # Load the model
    model = load_ann_model(resume=args.resume)
    model = model.to(device)

    # Initialise the loss function, adaptive optimizer, and epochs,
    # and macro F1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = args.epochs
    if (args.resume):
        checkpoint_model_dict = torch.load(
            ANN_MODEL_CHECKPOINT_PATH,
            map_location=device
        )
        start_epoch = checkpoint_model_dict["epoch"] + 1
        end_epoch = start_epoch + epochs
        macro_f1 = checkpoint_model_dict["macro_f1"]
        best_macro_f1 = checkpoint_model_dict["best_macro_f1"]
        optimizer.load_state_dict(
            checkpoint_model_dict["optimizer_state_dict"]
        )
    else:
        start_epoch = 0
        end_epoch = epochs
        macro_f1 = 0
        best_macro_f1 = float("-inf")

    # Train the model
    print(
        f"Starting training with the model at device: "
        f"{next(model.parameters()).device}"
    )
    for epoch in range(start_epoch, end_epoch):
        print("-----------------------------------------")
        print(f"Training epoch: {epoch + 1}")
        # Set the model to training mode
        model.train()
        
        # For every batch
        for X, y in train_loader: 
            # Move the training batch to device
            X = X.to(device)
            y = y.to(device)

            # Start this batch's training with the weights having 
            # zero gradients
            optimizer.zero_grad()

            # Do forward pass for a single batch of data into the
            # model
            logits = model(X)
            loss = loss_fn(logits, y)
            
            # Do backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

        # Validate the model
        macro_f1 = evaluate_ann_model(model, val_split)
        print(f"Validation macro F1: {macro_f1}")

        # If the model performs better during this validation,
        # update the best model
        if macro_f1 > best_macro_f1:
            print(f"***New best model found with macro F1: {macro_f1}***")
            best_macro_f1 = macro_f1
            update_best_model(
                model=model,
                epoch=epoch,
                macro_f1=macro_f1,
                best_macro_f1=best_macro_f1,
                optimizer=optimizer
            )

        # Update the checkpoint model
        update_checkpoint_model(
            model=model,
            epoch=epoch,
            macro_f1=macro_f1,
            best_macro_f1=best_macro_f1,
            optimizer=optimizer
        )
        print("-----------------------------------------")

    # Test the model
    print("-----------------------------------------")
    print("Testing model...")
    best_model_dict = torch.load(
        BEST_ANN_MODEL_PATH,
        map_location=device
    )
    best_model_state_dict = best_model_dict["model_state_dict"]
    model.load_state_dict(best_model_state_dict)
    macro_f1 = evaluate_ann_model(model, test_split)
    print(f"Macro F1: {macro_f1}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()