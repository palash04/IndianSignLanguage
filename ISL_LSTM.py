import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, p=0.5):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=1662, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Shape of x: (batch_size, frames, keypoints)
        output, (hidden, cell) = self.lstm(x)  # shape of hidden, cell: (num_layers, batch_size, hidden_size)
        hidden = hidden[-1]   # (batch_size, hidden_size)

        y = self.dropout(self.relu(self.fc1(hidden)))
        y = self.fc2(y)
        return y


def main():
    X = torch.rand((300, 30, 1662))
    model = RNN(10, num_layers=2)
    output = model(X)
    print(output.shape)


if __name__ == "__main__":
    main()
