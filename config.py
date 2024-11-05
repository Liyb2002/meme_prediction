import torch


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------- #

input_size = 3  # Number of input features (pnl, hold_length_hours, holding_percentage)
hidden_size = 64
num_classes = 2  # 'sell' or 'hold'
