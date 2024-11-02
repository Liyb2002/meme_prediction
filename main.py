import dataloader

import os
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Set the dataset directory
    current_dir = os.getcwd()
    dataset_directory = os.path.join(current_dir, 'dataset')  # Replace with your dataset path

    # Initialize the dataset
    dataset = dataloader.TradeDataset(dataset_directory)

    # Create the DataLoader
    batch_size = 16  # Adjust as needed
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataloader.custom_collate_fn
    )

    # Iterate over the DataLoader
    for batch_idx, (wallet_addresses, pnl_padded, hold_length_padded, holding_percentage_padded, seq_lengths) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Wallet Addresses:", len(wallet_addresses))
        print("PNL shape:", pnl_padded.shape)  # (batch_size, max_num_trades_in_batch)
        print("Hold Length Hours shape:", hold_length_padded.shape)
        print("Holding Percentage shape:", holding_percentage_padded.shape)
        print("Sequence Lengths:", seq_lengths)

        break