import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import re

class TradeDataset(Dataset):
    def __init__(self, dataset_directory):
        self.data = []
        self.load_data(dataset_directory)

    def load_data(self, dataset_directory):
        """
        Loads data from JSON files in the specified directory.
        """
        for filename in os.listdir(dataset_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(dataset_directory, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trades = data.get('trades', [])
                    for trade in trades:
                        wallet_address = trade.get('wallet_address')
                        pnl = trade.get('pnl')
                        hold_length = trade.get('hold_length')
                        holding_percentage = trade.get('holding_percentage')
                        
                        # Preprocess hold_length
                        total_minutes = self.parse_hold_length(hold_length)
                        
                        if None not in (wallet_address, pnl, total_minutes, holding_percentage):
                            self.data.append({
                                'wallet_address': wallet_address,
                                'pnl': pnl,
                                'hold_length_minutes': total_minutes,
                                'holding_percentage': holding_percentage
                            })

    def parse_hold_length(self, hold_length_str):
        """
        Parses the hold_length string (e.g., '2d 2h 17m') and converts it to total minutes.
        """
        try:
            days = hours = minutes = 0
            if 'd' in hold_length_str:
                days_match = re.search(r'(\d+)\s*d', hold_length_str)
                days = int(days_match.group(1)) if days_match else 0
            if 'h' in hold_length_str:
                hours_match = re.search(r'(\d+)\s*h', hold_length_str)
                hours = int(hours_match.group(1)) if hours_match else 0
            if 'm' in hold_length_str:
                minutes_match = re.search(r'(\d+)\s*m', hold_length_str)
                minutes = int(minutes_match.group(1)) if minutes_match else 0
            total_minutes = days * 1440 + hours * 60 + minutes
            return total_minutes
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
        """
        item = self.data[idx]
        wallet_address = item['wallet_address']
        pnl = torch.tensor(item['pnl'], dtype=torch.float32)
        hold_length_minutes = torch.tensor(item['hold_length_minutes'], dtype=torch.float32)
        holding_percentage = torch.tensor(item['holding_percentage'], dtype=torch.float32)
        return wallet_address, pnl, hold_length_minutes, holding_percentage


def custom_collate(batch):
    """
    Custom collate function to handle batching of data with string fields.
    """
    wallet_addresses = [item[0] for item in batch]
    pnls = torch.stack([item[1] for item in batch])
    hold_lengths = torch.stack([item[2] for item in batch])
    holding_percentages = torch.stack([item[3] for item in batch])
    return wallet_addresses, pnls, hold_lengths, holding_percentages
