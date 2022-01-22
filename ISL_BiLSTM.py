import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, bidirectional=True, p=0.5):
        super(BiLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=1662,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            bidirectional=bidirectional)

        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, 64)
        else:
            self.fc1 = nn.Linear(hidden_size, 64)

        self.fc2 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Shape of x: (batch_size, frames, keypoints)
        batch_size, _, _ = x.shape
        x = x.permute(1, 0, 2)  # (frames, batch_size, keypoints)
        _, (hidden, cell) = self.lstm(x)
        # shape of hidden, cell: (num_layers * num_directions, batch_size, hidden_size)

        if self.bidirectional:
            hidden = hidden.reshape(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = hidden[-1]
        else:
            hidden = hidden[-1].unsqueeze(0)

        hidden = hidden.reshape(batch_size, -1)

        out = self.dropout(self.relu(self.fc1(hidden)))
        out = self.fc2(out)
        return out  # (batch_size, num_classes)


def main():
    X = torch.zeros((40, 5, 1662))
    target = torch.zeros(40)
    num_classes = 10
    model = BiLSTM(num_classes, num_layers=2, bidirectional=True)

    output = model(X.float())    # (batch_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target.long())
    print(loss.item())


if __name__ == "__main__":
    main()
