import os
import json
import torch
import re
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TradeDataset(Dataset):
    def __init__(self, dataset_directory):
        self.data = []
        self.load_data(dataset_directory)

    def load_data(self, dataset_directory):
        """
        Loads data from JSON files in the specified directory.
        Each JSON file corresponds to a wallet_address.
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

                    # Initialize lists to store variables
                    pnl_list = []
                    hold_length_list = []
                    holding_percentage_list = []

                    for trade in trades:
                        # Verify that the wallet_address matches
                        if trade.get('wallet_address') != wallet_address:
                            continue  # Skip inconsistent entries
                        pnl = trade.get('pnl')
                        hold_length = trade.get('hold_length')
                        holding_percentage = trade.get('holding_percentage')
                        if None in (pnl, hold_length, holding_percentage):
                            continue  # Skip trades with missing data

                        # Convert hold_length to a numerical value (e.g., hours)
                        hold_length_hours = self.extract_hold_length_hours(hold_length)
                        if hold_length_hours is None:
                            continue  # Skip if hold_length couldn't be parsed

                        # Append variables to the lists
                        pnl_list.append(float(pnl))
                        hold_length_list.append(float(hold_length_hours))
                        holding_percentage_list.append(float(holding_percentage))

                    if pnl_list and hold_length_list and holding_percentage_list:
                        # All lists should be of the same length
                        self.data.append({
                            'wallet_address': wallet_address,
                            'pnl': pnl_list,
                            'hold_length_hours': hold_length_list,
                            'holding_percentage': holding_percentage_list
                        })

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
        Returns the total number of wallet addresses.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the data for the wallet address at the specified index.
        """
        item = self.data[idx]
        wallet_address = item['wallet_address']
        pnl = torch.tensor(item['pnl'], dtype=torch.float32)
        hold_length_hours = torch.tensor(item['hold_length_hours'], dtype=torch.float32)
        holding_percentage = torch.tensor(item['holding_percentage'], dtype=torch.float32)
        return wallet_address, pnl, hold_length_hours, holding_percentage


def custom_collate_fn(batch):
    """
    Custom collate function to batch data.

    Args:
        batch: List of tuples returned by __getitem__:
            - wallet_address: str
            - pnl: Tensor of shape (num_trades,)
            - hold_length_hours: Tensor of shape (num_trades,)
            - holding_percentage: Tensor of shape (num_trades,)

    Returns:
        batch_wallet_addresses: List[str]
        batch_pnl: Tensor of shape (batch_size, num_trades)
        batch_hold_length_hours: Tensor of shape (batch_size, num_trades)
        batch_holding_percentage: Tensor of shape (batch_size, num_trades)
    """
    batch_wallet_addresses = [item[0] for item in batch]
    pnl_list = [item[1] for item in batch]
    hold_length_list = [item[2] for item in batch]
    holding_percentage_list = [item[3] for item in batch]

    # Get sequence lengths
    seq_lengths = torch.tensor([len(pnl) for pnl in pnl_list], dtype=torch.long)

    # Pad sequences
    batch_pnl_padded = pad_sequence(pnl_list, batch_first=True, padding_value=0.0)
    batch_hold_length_hours_padded = pad_sequence(hold_length_list, batch_first=True, padding_value=0.0)
    batch_holding_percentage_padded = pad_sequence(holding_percentage_list, batch_first=True, padding_value=0.0)

    return (batch_wallet_addresses,
            batch_pnl_padded,
            batch_hold_length_hours_padded,
            batch_holding_percentage_padded,
            seq_lengths)
