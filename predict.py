import torch
import torch.nn as nn
import os

import model
from config import device, input_size, hidden_size, num_classes



def load_models():
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    trade_model = model.TradeClassifier(input_size, hidden_size, num_classes).to(device)

    trade_model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth'), weights_only=True))

    return trade_model


def predict(state, trade_model):
    pnl, hold_length_hours, holding_percentage = state
    features = torch.tensor([pnl, hold_length_hours, holding_percentage], dtype=torch.float32)
    features = features.unsqueeze(0).to(device)  # Shape: (1, 3)

    with torch.no_grad():
        outputs = trade_model(features)
        probabilities = nn.functional.softmax(outputs, dim=1)
        
        probabilities = probabilities.cpu().numpy()[0]
    return probabilities  # [prob_hold, prob_sell]



if __name__ == '__main__':
    trade_model = load_models()

    # (pnl, hold_length_hours, holding_percentage)
    state = (2.0, 24.0, 0.75)

    probabilities = predict(state, trade_model)

    print(f"Probabilities: Hold = {probabilities[0]:.4f}, Sell = {probabilities[1]:.4f}")
