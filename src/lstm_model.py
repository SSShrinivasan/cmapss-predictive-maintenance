import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])

        x = self.fc(x)
        return x