import torch.nn as nn
import torch.optim as optim

class TradeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TradeClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out
