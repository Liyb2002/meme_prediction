import os
import json
from torch.utils.data import Dataset, DataLoader


class TradePNLDataset(Dataset):
    def __init__(self, dataset_directory):
        """
        Initializes the dataset by loading all pnl data from JSON files.

        :param dataset_directory: Path to the directory containing JSON files.
        """
        self.pnl_data = []
        self.load_data(dataset_directory)

    def load_data(self, dataset_directory):
        """
        Loads pnl data from all JSON files in the dataset directory.

        :param dataset_directory: Path to the directory containing JSON files.
        """
        for filename in os.listdir(dataset_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(dataset_directory, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trades = data.get('trades', [])
                    for trade in trades:
                        pnl = trade.get('pnl', None)
                        if pnl is not None:
                            self.pnl_data.append(pnl)

    def __len__(self):
        return len(self.pnl_data)

    def __getitem__(self, idx):
        return self.pnl_data[idx]
