import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # Output size matches input_size

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_in_last_timestep = h_n[-1, :, :]
        x = self.fc(output_in_last_timestep)
        return x
if __name__ == "__main__":
    # Example usage
    input_size = 200  # Number of features
    model = LSTMModel(input_size=input_size)

    # Example input
    # batch_size = 32, sequence_length = 20
    x = torch.randn(32, 20, input_size)
    output = model(x)
    print(output.shape)  # Should be (32, 200)
