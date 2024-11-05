import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import re
from torch.nn.utils.rnn import pad_sequence

class TradeDataset(Dataset):
    def __init__(self, dataset_directory):
        self.data = []
        self.load_data(dataset_directory)

    def load_data(self, dataset_directory):
        """
        Loads data from JSON files in the specified directory.
        For each 'sell' data point, generates a 'hold' data point.
        Each data point is an individual state.
        """
        for filename in os.listdir(dataset_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(dataset_directory, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trades = data.get('trades', [])
                    if not trades:
                        continue  # Skip files with no trades
                    # Since all trades in a file have the same wallet_address
                    wallet_address = trades[0].get('wallet_address')
                    if wallet_address is None:
                        continue  # Skip if wallet_address is missing

                    # Find scaling factor using wallet_address
                    scaling_factor = self.find_random(wallet_address)
                    if scaling_factor is None:
                        continue  # Skip if scaling factor couldn't be determined

                    for trade in trades:
                        # Extract variables
                        pnl = trade.get('pnl')
                        hold_length = trade.get('hold_length')
                        holding_percentage = trade.get('holding_percentage')
                        if None in (pnl, hold_length, holding_percentage):
                            continue  # Skip trades with missing data

                        # Convert hold_length to numerical value
                        hold_length_hours = self.extract_hold_length_hours(hold_length)
                        if hold_length_hours is None:
                            continue  # Skip if hold_length couldn't be parsed

                        # Create 'sell' data point
                        sell_data = {
                            'wallet_address': wallet_address,
                            'pnl': float(pnl),
                            'hold_length_hours': float(hold_length_hours),
                            'holding_percentage': float(holding_percentage),
                            'label': 1  # 'sell' label
                        }
                        self.data.append(sell_data)

                        # Generate 'hold' data point using the scaling factor and new calculations
                        scaled_hold_length_factor = (scaling_factor * 2) % 1
                        scaled_holding_percentage_factor = (scaling_factor * 4) % 1

                        hold_data = {
                            'wallet_address': wallet_address,
                            'pnl': float(pnl) * scaling_factor,
                            'hold_length_hours': float(hold_length_hours) * scaled_hold_length_factor,
                            'holding_percentage': float(holding_percentage) * scaled_holding_percentage_factor,
                            'label': 0  # 'hold' label
                        }
                        self.data.append(hold_data)

    def find_random(self, wallet_address):
        """
        Extracts the first two valid digits from the wallet_address and returns
        a scaling factor by multiplying the number by 0.01.

        Args:
            wallet_address (str): The wallet address string.

        Returns:
            float: The scaling factor between 0 and 1, or None if not found.
        """
        digits = ''
        for char in wallet_address:
            if char.isdigit():
                digits += char
                if len(digits) == 2:
                    break

        if len(digits) == 2:
            scaling_factor = int(digits) * 0.01
        elif len(digits) == 1:
            scaling_factor = int(digits) * 0.01
        else:
            print(f"Warning: No digits found in wallet_address '{wallet_address}'.")
            return None  # Or set a default scaling factor if desired

        # Ensure the scaling factor is between 0 and 1
        scaling_factor = max(0.01, min(scaling_factor, 1.0))

        return scaling_factor

    def extract_hold_length_hours(self, hold_length_str):
        """
        Extracts the number from a hold_length string like '29h' and returns it as a float.
        """
        try:
            # Use regular expression to extract the numeric part
            match = re.match(r'(\d+\.?\d*)h', hold_length_str)
            if match:
                hours = float(match.group(1))
                return hours
            else:
                print(f"Warning: Could not parse hold_length '{hold_length_str}'. Setting to None.")
                return None
        except Exception as e:
            print(f"Error parsing hold_length '{hold_length_str}': {e}")
            return None

    def __len__(self):
        """
        Returns the total number of data points.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the data point at the specified index.
        Each data point is an individual state.
        """
        item = self.data[idx]
        # Since wallet_address is not a feature, we can ignore it or return it for reference
        pnl = torch.tensor(item['pnl'], dtype=torch.float32)
        hold_length_hours = torch.tensor(item['hold_length_hours'], dtype=torch.float32)
        holding_percentage = torch.tensor(item['holding_percentage'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)  # Use long for classification labels
        return pnl, hold_length_hours, holding_percentage, label




# --------------------- custom_collate_fn --------------------- #

def custom_collate_fn(batch):
    """
    Custom collate function to batch data points.
    Each data point is a tuple of (pnl, hold_length_hours, holding_percentage, label).
    """
    pnl_list = torch.tensor([item[0] for item in batch], dtype=torch.float32)
    hold_length_list = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    holding_percentage_list = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    labels_tensor = torch.tensor([item[3] for item in batch], dtype=torch.long)

    # Stack features into a single tensor
    features_tensor = torch.stack([pnl_list, hold_length_list, holding_percentage_list], dim=1)
    # features_tensor shape: (batch_size, 3)

    return features_tensor, labels_tensor
