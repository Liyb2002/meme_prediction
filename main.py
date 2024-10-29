import dataloader

import os
import json
from torch.utils.data import Dataset, DataLoader

# Example usage:
if __name__ == '__main__':
    current_dir = os.getcwd()
    dataset_directory = os.path.join(current_dir, 'dataset')  # Corrected line
    batch_size = 32  # Adjust batch size as needed

    # Create the dataset and dataloader
    dataset = dataloader.TradePNLDataset(dataset_directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the DataLoader
    for batch_idx, pnl_batch in enumerate(dataloader):
        # pnl_batch is a tensor containing a batch of pnl values
        print(f"Batch {batch_idx + 1}:")
        print(pnl_batch)
        # Here you can add your training code, e.g., feed pnl_batch into your model
