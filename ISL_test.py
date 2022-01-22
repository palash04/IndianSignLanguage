import torch
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from ISL_BiLSTM import BiLSTM
from ISL_LSTM import RNN
from ISL_Transformer import TransformerEncoder
from ISL_preprocess_keypts import get_sequences_labels
from ISL_params import *
from ISL_utils import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def test_model(model, data_loader, device):
    model.eval()

    predictions_list = []
    true_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
            x, y = x.double(), y.double()
            x = x.to(device)  # (batch_size, frames, keypoints)
            y = y.to(device)  # (batch_size)

            output = model(x)  # (batch_size, num_classes)

            _, preds = torch.max(output, dim=1)

            predictions_list.append(preds.view(-1).cpu())
            true_list.append(y.view(-1).cpu())

    return predictions_list, true_list


def main():
    parser = argparse.ArgumentParser(description='ISL')
    parser.add_argument('--gpuidx', type=int, default=0, help='GPU index (default: 0)')

    args = parser.parse_args()
    gpuidx = args.gpuidx

    device = f'cuda:{gpuidx}' if torch.cuda.is_available() else 'cpu'
    data_dir = test_dir
    sequences, labels = get_sequences_labels(ROOT_DIR=data_dir)

    X = torch.tensor(sequences)
    y = torch.tensor(labels)

    dataset = TensorDataset(X, y)

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    num_classes = len(labels_to_gestures)
    if MODEL_NAME == 'lstm':
        model = RNN(num_classes, hidden_size=HIDDEN_SIZE)
    elif MODEL_NAME == 'bilstm':
        model = BiLSTM(num_classes, hidden_size=HIDDEN_SIZE, bidirectional=BIDIRECTIONAL)
    else:
        model = TransformerEncoder(num_classes, hidden_size=HIDDEN_SIZE, device=device)

    model = model.double().to(device)
    model.load_state_dict(torch.load(f'best_model_{MODEL_NAME}.pth.tar'))

    predictions_list, true_list = test_model(model, test_loader, device)

    y_pred = torch.cat(predictions_list).numpy()
    y_true = torch.cat(true_list).numpy()

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Test dataset accuracy: {acc}')

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=gestures))

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plot_cm_heatmap(cm)


if __name__ == "__main__":
    main()
