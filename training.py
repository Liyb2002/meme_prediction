import meme_dataloader 
import model
from config import device, input_size, hidden_size, num_classes
import helper

import os
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def prepare_data(dataset):

    # For user_embedding
    addr_to_trades = {}
    for idx in tqdm(range(len(dataset)), desc=f"Preparing Data - User Embedding"):
        addr, pnl, hold_length_hours, holding_percentage, label = dataset[idx]
        if label == 0:
            continue

        if addr not in addr_to_trades:
            addr_to_trades[addr] = []
        
        addr_to_trades[addr].append(pnl.item())
        addr_to_trades[addr].append(hold_length_hours.item())
        addr_to_trades[addr].append(holding_percentage.item())
        addr_to_trades[addr].append(0)

    addr_to_trades = helper.padd_trade_map(addr_to_trades)


    full_data_list = []
    labels_list = []

    for idx in tqdm(range(len(dataset)), desc=f"Preparing Data - Logic Model"):
        addr, pnl, hold_length_hours, holding_percentage, label = dataset[idx]
        if addr not in addr_to_trades:
            continue
        
        user_embedding = addr_to_trades[addr]
        full_data = user_embedding + [pnl.item(), hold_length_hours.item(), holding_percentage.item()]

        full_data_list.append(full_data)
        labels_list.append(label)


    features_tensor = torch.tensor(full_data_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

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
    train_tensor_dataset = TensorDataset( train_features, train_labels)
    val_tensor_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32  # Adjust as needed
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)


    state_embedding_model = model.StateEmbeddingNetwork().to(device)
    user_embedding_model = model.UserEmbeddingNetwork().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(state_embedding_model.parameters()) + list(user_embedding_model.parameters()), lr=0.001)

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)


    num_epochs = 30
    best_val_loss = 100
    for epoch in range(num_epochs):
        state_embedding_model.train()
        user_embedding_model.train()

        total_train_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()

            print("features", features.shape)
            print('labels', labels.shape)














            # Forward pass
            user_embedding = user_embedding_model(train_addr_to_trades_tensor)

            print("user_embedding", user_embedding.shape)
            state_embedding = state_embedding_model(features)
            print("state_embedding", state_embedding.shape)

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
            for val_addr_to_trades_tensor, features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
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