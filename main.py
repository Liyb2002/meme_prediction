import dataloader

import os
import torch
from torch.utils.data import DataLoader

# Assume TradeDataset and custom_collate have been defined as in the previous code
# Here's the main code that reads the dataset

if __name__ == '__main__':
    # Set the dataset directory
    current_dir = os.getcwd()
    dataset_directory = os.path.join(current_dir, 'dataset')  # Replace with your dataset path if different

    # Initialize the dataset
    dataset = dataloader.TradeDataset(dataset_directory)

    # Create the DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=32,        # Adjust batch size as needed
        shuffle=True,         # Shuffle data at every epoch
        collate_fn=dataloader.custom_collate  # Use the custom collate function
    )

    # Iterate over the DataLoader
    for batch_idx, (wallet_addresses, pnls, hold_lengths, holding_percentages) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print("Wallet Addresses:", wallet_addresses)
        print("PNLs:", pnls)
        print("Hold Lengths (minutes):", hold_lengths)
        print("Holding Percentages:", holding_percentages)

        # Break after one batch for demonstration purposes
        # Remove this break if you want to process all batches
        break
