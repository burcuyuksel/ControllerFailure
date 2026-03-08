import torch
from lstm_train import model  # Assuming `model` is defined in `lstm_train.py`

torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved as lstm_model.pth")
