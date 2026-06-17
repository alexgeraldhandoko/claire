from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_DIR))

from scripts.helper.constants import (
    TRAINING_EPOCHS, 
    device
)
from scripts.helper.helper_classes import TensorDatasetSplit, LoadedTensors
from scripts.helper.helper_functions import (
    load_tensors, 
    build_ann_model, 
    evaluate_model,
    update_best_model,
    update_checkpoint_model,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

def main():
    # Print device type
    print(f"Training is using device: {device}")

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

    # Build the model
    print("Building the model...")
    model = build_ann_model().to(device=device)

    # Initialise the loss function and adaptive optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training")
    best_macro_f1 = 0
    for epoch in range(TRAINING_EPOCHS):
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
        macro_f1 = evaluate_model(model, val_split)
        print(f"Validation macro F1: {macro_f1}")

        # If the model performs better during this validation,
        # update the best model
        if macro_f1 > best_macro_f1:
            print(f"***New best model found with macro F1: {macro_f1}***")
            best_macro_f1 = macro_f1
            update_best_model(model, best_macro_f1)

        # Update the checkpoint model
        update_checkpoint_model(model, macro_f1)
        print("-----------------------------------------")

    # Test the model
    print("-----------------------------------------")
    print("Testing model...")
    macro_f1 = evaluate_model(model, test_split)
    print(f"Macro F1: {macro_f1}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()