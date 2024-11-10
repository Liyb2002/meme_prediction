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
        # Remove this line to include all labels
        # if label == 0:
        #     continue

        if addr not in addr_to_trades:
            addr_to_trades[addr] = []
        
        addr_to_trades[addr].append(pnl.item())
        addr_to_trades[addr].append(hold_length_hours.item())
        addr_to_trades[addr].append(holding_percentage.item())
        addr_to_trades[addr].append(0)  # You can adjust this if needed

    # Pad or process addr_to_trades as required
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
        labels_list.append(float(label))  # Ensure labels are floats (0.0 or 1.0)

    features_tensor = torch.tensor(full_data_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float32)  # Use torch.float32

    return features_tensor, labels_tensor



def compute_accuracy(logits, labels):
    probabilities = torch.sigmoid(logits)  # Shape: (batch_size, 1) or (batch_size,)

    predictions = (probabilities >= 0.5).float()  # Shape: (batch_size, 1) or (batch_size,)

    predictions = predictions.view(-1)  # Shape: (batch_size,)
    labels = labels.view(-1)  # Shape: (batch_size,)

    # Compute number of correct predictions
    correct_predictions = (predictions == labels).sum().item()

    # Total number of samples in the batch
    total_in_batch = labels.size(0)

    return total_in_batch, correct_predictions




def load_models(state_embedding_model, user_embedding_model, cross_attention_model, save_dir):
    state_embedding_model.load_state_dict(torch.load(os.path.join(state_embedding_model, 'state_embedding_model.pth')))
    state_embedding_model.to(device)
    state_embedding_model.eval()

    user_embedding_model.load_state_dict(torch.load(os.path.join(user_embedding_model, 'user_embedding_model.pth')))
    user_embedding_model.to(device)
    user_embedding_model.eval()

    cross_attention_model.load_state_dict(torch.load(os.path.join(cross_attention_model, 'cross_attention_model.pth')))
    cross_attention_model.to(device)
    cross_attention_model.eval()


def save_models(state_embedding_model, user_embedding_model, cross_attention_model, save_dir):
    torch.save(state_embedding_model.state_dict(), os.path.join(save_dir, 'state_embedding_model.pth'))
    torch.save(user_embedding_model.state_dict(), os.path.join(save_dir, 'user_embedding_model.pth'))
    torch.save(cross_attention_model.state_dict(), os.path.join(save_dir, 'cross_attention_model.pth'))



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
    cross_attention_model = model.UserStateCrossAttentionModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
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

        train_total_correct = 0
        train_total_samples = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()

            user_embedding = user_embedding_model(features)
            state_embedding = state_embedding_model(features)

            # print("user_embedding", user_embedding.shape)
            # print("state_embedding", state_embedding.shape)

            logits = cross_attention_model(state_embedding, user_embedding)
            # print("logits", logits.shape)
            # print("labels", labels.shape)


            # Compute loss
            loss = criterion(logits.squeeze(), labels)  # Squeeze logits to match labels shape

            total_in_batch, correct_predictions = compute_accuracy(logits, labels)
            train_total_correct += correct_predictions
            train_total_samples += total_in_batch

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        accuracy = train_total_correct / train_total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {accuracy:.4f}")

        # Validation phase
        state_embedding_model.eval()
        user_embedding_model.eval()
        cross_attention_model.eval()

        total_val_loss = 0
        val_total_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):

                # Forward pass
                user_embedding = user_embedding_model(features)
                state_embedding = state_embedding_model(features)
                logits = cross_attention_model(state_embedding, user_embedding)


                # Compute loss
                loss = criterion(logits.squeeze(), labels)  # Squeeze logits to match labels shape

                total_in_batch, correct_predictions = compute_accuracy(logits, labels)
                val_total_correct += correct_predictions
                val_total_samples += total_in_batch

        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_total_correct / val_total_samples

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Val Loss: {average_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
        
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_models(state_embedding_model, user_embedding_model, cross_attention_model, save_dir)
            print("save model!")