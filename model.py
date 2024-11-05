import torch.nn as nn
import torch.optim as optim

class StateEmbeddingNetwork(nn.Module):
    def __init__(self, state_feature_size = 3, state_embedding_size = 8):
        super(StateEmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_feature_size, state_embedding_size),
            nn.ReLU()
        )

    def forward(self, state_features):
        # state_features: Tensor of shape (batch_size, state_feature_size)
        state_embeddings = self.fc(state_features)  # Shape: (batch_size, state_embedding_size)
        return state_embeddings



class UserEmbeddingNetwork(nn.Module):
    def __init__(self, input_size = 3, hidden_size = 64, embedding_size = 32, num_layers=1):
        super(UserEmbeddingNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, trade_sequences, sequence_lengths):
        # Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(trade_sequences, sequence_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # Use the final hidden state as the user embedding
        embedding = self.fc(h_n[-1])
        return embedding  # Shape: (batch_size, embedding_size)
