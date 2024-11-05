import meme_dataloader 
import model
from config import device, input_size, hidden_size, num_classes

import os
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def prepare_data(dataset):
    pnl_list = []
    hold_length_list = []
    holding_percentage_list = []
    labels_list = []
    for idx in tqdm(range(len(dataset)), desc=f"Preparing Data"):
        pnl, hold_length_hours, holding_percentage, label = dataset[idx]
        pnl_list.append(pnl)
        hold_length_list.append(hold_length_hours)
        holding_percentage_list.append(holding_percentage)
        labels_list.append(label)

    # Stack features and labels
    pnl_tensor = torch.stack(pnl_list)
    hold_length_tensor = torch.stack(hold_length_list)
    holding_percentage_tensor = torch.stack(holding_percentage_list)
    labels_tensor = torch.stack(labels_list)

    # Create the input feature tensor
    features_tensor = torch.stack([pnl_tensor, hold_length_tensor, holding_percentage_tensor], dim=1)

    return features_tensor, labels_tensor



def load_models(model, save_dir):
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth')))
    model.to(device)
    model.eval


def save_models(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))



if __name__ == '__main__':
    # Set the dataset directory
    current_dir = os.getcwd()
    dataset_directory = os.path.join(current_dir, 'dataset')  # Replace with your dataset path

    # Initialize the dataset
    dataset = meme_dataloader.TradeDataset(dataset_directory)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_features, train_labels = prepare_data(train_dataset)
    # Prepare validation data
    val_features, val_labels = prepare_data(val_dataset)

    # Move data to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    # Create TensorDatasets
    train_tensor_dataset = TensorDataset(train_features, train_labels)
    val_tensor_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32  # Adjust as needed
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)


    model = model.TradeClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)


    num_epochs = 30
    best_val_loss = 100
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # features: Tensor of shape (batch_size, 3)
                # labels: Tensor of shape (batch_size,)

                # Forward pass
                outputs = model(features)

                # Compute loss
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {average_train_loss:.4f}, '
              f'Val Loss: {average_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
        
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_models(model, save_dir)
            print("save model!")