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

                    pnl_list = []
                    hold_length_list = []
                    holding_percentage_list = []

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

                        # Append original 'sell' data
                        pnl_list.append(float(pnl))
                        hold_length_list.append(float(hold_length_hours))
                        holding_percentage_list.append(float(holding_percentage))

                    if not pnl_list:
                        continue  # Skip if no valid trades found

                    # Create 'sell' data point
                    sell_data = {
                        'wallet_address': wallet_address,
                        'pnl': pnl_list,
                        'hold_length_hours': hold_length_list,
                        'holding_percentage': holding_percentage_list,
                        'label': 1  # 'sell' label
                    }
                    self.data.append(sell_data)

                    # Generate 'hold' data point using the scaling factors
                    scaling_factor_pnl = scaling_factor
                    scaling_factor_hold_length = (scaling_factor * 2) % 1
                    scaling_factor_holding_percentage = (scaling_factor * 4) % 1

                    hold_pnl_list = [p * scaling_factor_pnl for p in pnl_list]
                    hold_hold_length_list = [h * scaling_factor_hold_length for h in hold_length_list]
                    hold_holding_percentage_list = [hp * scaling_factor_holding_percentage for hp in holding_percentage_list]

                    hold_data = {
                        'wallet_address': wallet_address,
                        'pnl': hold_pnl_list,
                        'hold_length_hours': hold_hold_length_list,
                        'holding_percentage': hold_holding_percentage_list,
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
        """
        item = self.data[idx]
        wallet_address = item['wallet_address']
        pnl = torch.tensor(item['pnl'], dtype=torch.float32)
        hold_length_hours = torch.tensor(item['hold_length_hours'], dtype=torch.float32)
        holding_percentage = torch.tensor(item['holding_percentage'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)  # Use long for classification labels
        return wallet_address, pnl, hold_length_hours, holding_percentage, label




# --------------------- custom_collate_fn --------------------- #

def custom_collate_fn(batch):
    """
    Custom collate function to batch data with variable-length sequences.
    Handles padding of sequences.

    Args:
        batch: List of tuples returned by __getitem__:
            - wallet_address: str
            - pnl: Tensor of shape (sequence_length,)
            - hold_length_hours: Tensor of shape (sequence_length,)
            - holding_percentage: Tensor of shape (sequence_length,)
            - label: Tensor of shape ()

    Returns:
        batch_wallet_addresses: List[str]
        batch_features_padded: Tensor of shape (batch_size, max_seq_length, 3)
        batch_labels: Tensor of shape (batch_size,)
        seq_lengths: Tensor of shape (batch_size,)
    """
    batch_wallet_addresses = [item[0] for item in batch]
    pnl_list = [item[1] for item in batch]
    hold_length_list = [item[2] for item in batch]
    holding_percentage_list = [item[3] for item in batch]
    batch_labels = torch.stack([item[4] for item in batch])

    # Get sequence lengths
    seq_lengths = torch.tensor([len(pnl) for pnl in pnl_list], dtype=torch.long)

    # Handle empty sequences
    for i in range(len(pnl_list)):
        if seq_lengths[i] == 0:
            # Replace empty sequences with a tensor containing a single zero
            pnl_list[i] = torch.tensor([0.0], dtype=torch.float32)
            hold_length_list[i] = torch.tensor([0.0], dtype=torch.float32)
            holding_percentage_list[i] = torch.tensor([0.0], dtype=torch.float32)
            seq_lengths[i] = 1  # Update sequence length

    # Pad sequences
    batch_pnl_padded = pad_sequence(pnl_list, batch_first=True, padding_value=0.0)
    batch_hold_length_padded = pad_sequence(hold_length_list, batch_first=True, padding_value=0.0)
    batch_holding_percentage_padded = pad_sequence(holding_percentage_list, batch_first=True, padding_value=0.0)

    # Stack features into a single tensor
    batch_features_padded = torch.stack(
        [batch_pnl_padded, batch_hold_length_padded, batch_holding_percentage_padded], dim=2
    )
    # batch_features_padded shape: (batch_size, max_seq_length, 3)

    return batch_wallet_addresses, batch_features_padded, batch_labels, seq_lengths
