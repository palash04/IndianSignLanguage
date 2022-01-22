import torch
import torch.nn as nn
from ISL_params import *

class TransformerEncoder(nn.Module):
    def __init__(self, num_classes, device, hidden_size=128, p=0.5):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.pos_embeddings = nn.Embedding(FRAMES_PER_VIDEO, 1662)
        self.transformer = nn.TransformerEncoderLayer(d_model=1662, nhead=6)  # nhead has to be divisible by 1662

        self.fc1 = nn.Linear(1662, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Shape of x: (batch_size, frames, keypoints)
        batch_size, frames, _ = x.shape

        positions = torch.arange(0, frames).expand(batch_size, frames).to(self.device)  # (batch_size, frames)
        pos_embed = self.pos_embeddings(positions)  # (batch_size, frames, keypoints)

        embed_x = x + pos_embed  # (batch_size, frames, keypoints)

        embed_x = embed_x.permute(1, 0, 2)  # (frames, batch_size, keypoints)

        trans_out = self.transformer(embed_x)  # (frames, batch_size, keypoints)
        trans_out = trans_out.mean(0)  # (batch_size, keypoints)

        out = self.dropout(self.relu(self.fc1(trans_out)))  # (batch_size, 128)
        out = self.fc2(out)  # (batch_size, num_classes)
        return out


def main():
    X = torch.zeros((40, 5, 1662))
    target = torch.zeros(40)
    num_classes = 10
    model = TransformerEncoder(num_classes, device='cpu')

    output = model(X.float())  # (batch_size, num_classes)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, target.long())
    print(loss.item())




if __name__ == "__main__":
    main()
