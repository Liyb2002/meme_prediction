import meme_dataloader 

import os
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Set the dataset directory
    current_dir = os.getcwd()
    dataset_directory = os.path.join(current_dir, 'dataset')  # Replace with your dataset path

    # Initialize the dataset
    dataset = meme_dataloader.TradeDataset(dataset_directory)

    # Create the DataLoader
    batch_size = 32  # Adjust as needed
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=meme_dataloader.custom_collate_fn
    )

    # Iterate over the DataLoader
    for batch_idx, (wallet_addresses, features_padded, labels, seq_lengths) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Wallet Addresses:", wallet_addresses)
        print("Features shape:", features_padded.shape)  # (batch_size, max_seq_length, 3)
        print("Labels:", labels)
        print("Sequence Lengths:", seq_lengths)
        
        break