import torch.nn as nn
import torch.optim as optim
import torch

class StateEmbeddingNetwork(nn.Module):
    def __init__(self, input_size=3, embedding_size=16):
        super(StateEmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )

    def forward(self, features):
        # Extract the last 3 values
        state_features = features[:, -3:]  # Shape: (batch_size, 3)
        embedding = self.fc(state_features)  # Shape: (batch_size, embedding_size)
        return embedding






class UserEmbeddingNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, embedding_size=32, num_layers=1):
        super(UserEmbeddingNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,   # Number of features per trade
            hidden_size=hidden_size, # Hidden state size
            num_layers=num_layers,   # Number of LSTM layers
            batch_first=True         # Batch size is the first dimension
        )
        self.fc = nn.Linear(hidden_size, embedding_size)  # Maps LSTM output to embedding size

    def forward(self, features):
        # features: (batch_size, 83)
        batch_size = features.size(0)
        seq_len = 20
        input_size = 4
        # Reshape the first 80 features
        trade_sequences = features[:, :80].view(batch_size, seq_len, input_size)
        # Pass through LSTM
        output, (h_n, c_n) = self.lstm(trade_sequences)
        # Get the last hidden state
        embedding = self.fc(h_n[-1])  # Shape: (batch_size, embedding_size)
        return embedding





class UserStateCrossAttentionModel(nn.Module):
    def __init__(self, query_dim=16, key_value_dim=32, num_heads=4, hidden_size=64):
        super(UserStateCrossAttentionModel, self).__init__()
        # Projection layers for query, key, and value
        self.query_proj = nn.Linear(query_dim, key_value_dim)
        self.key_proj = nn.Linear(key_value_dim, key_value_dim)
        self.value_proj = nn.Linear(key_value_dim, key_value_dim)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=key_value_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Decoder (classifier) for binary prediction
        self.decoder = nn.Sequential(
            nn.Linear(key_value_dim + query_dim, hidden_size),  # Updated input dimension
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state_embedding, user_embedding):
        """
        Performs cross-attention between state_embedding and user_embedding,
        concatenates the attention output with state_embedding,
        and decodes the result to produce a binary prediction.

        Args:
            state_embedding (Tensor): Tensor of shape (batch_size, query_dim).
            user_embedding (Tensor): Tensor of shape (batch_size, key_value_dim).

        Returns:
            Tensor: Logits of shape (batch_size, 1).
        """
        # Project state_embedding to query
        query = self.query_proj(state_embedding).unsqueeze(1)  # Shape: (batch_size, 1, key_value_dim)

        # Project user_embedding to key and value
        key = self.key_proj(user_embedding).unsqueeze(1)    # Shape: (batch_size, 1, key_value_dim)
        value = self.value_proj(user_embedding).unsqueeze(1)  # Shape: (batch_size, 1, key_value_dim)

        # Perform attention
        attn_output, _ = self.attention(query, key, value)  # attn_output: (batch_size, 1, key_value_dim)

        # Remove the sequence dimension
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, key_value_dim)

        # Concatenate attention output with state_embedding
        combined = torch.cat((attn_output, state_embedding), dim=1)  # Shape: (batch_size, key_value_dim + query_dim)

        # Pass through the decoder to get logits
        logits = self.decoder(combined)  # Shape: (batch_size, 1)

        return logits
