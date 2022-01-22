import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ISL_LSTM import RNN
from ISL_BiLSTM import BiLSTM
from ISL_Transformer import TransformerEncoder
from ISL_preprocess_keypts import get_sequences_labels
from ISL_params import *
from ISL_utils import *


def train_epoch(model, data_loader, device, criterion, optimizer, scheduler):
    model.train()

    losses = []
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.double(), y.double()
        x = x.to(device)  # (batch_size, frames, keypoints)
        y = y.to(device)  # (batch_size)

        output = model(x)  # (batch_size, num_classes)

        loss = criterion(output, y.long())

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        preds = F.softmax(output, dim=1)
        preds = preds.argmax(dim=1, keepdim=True).reshape(-1)
        correct += (preds.long() == y.long()).sum().item()
        total += preds.size(0)

    acc = (correct * 1.0) / total
    # scheduler.step(acc)

    return acc, np.mean(losses)


def val_epoch(model, data_loader, device, criterion):
    model.eval()

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.double(), y.double()
            x = x.to(device)  # (batch_size, frames, keypoints)
            y = y.to(device)  # (batch_size)

            output = model(x)  # (batch_size, num_classes)

            loss = criterion(output, y.long())

            losses.append(loss.item())

            preds = F.softmax(output, dim=1)
            preds = preds.argmax(dim=1, keepdim=True).reshape(-1)
            correct += (preds.long() == y.long()).sum().item()
            total += preds.size(0)

    acc = (correct * 1.0) / total

    return acc, np.mean(losses)


def train(model, epochs, device, train_loader, val_loader, criterion, optimizer, scheduler):
    best_val_acc = 0

    for _ in tqdm(range(epochs)):
        _, _ = train_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        val_acc, _ = val_epoch(model, val_loader, device, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='ISL')
    parser.add_argument('--gpuidx', type=int, default=0, help='GPU index (default: 0)')

    args = parser.parse_args()
    gpuidx = args.gpuidx

    device = f'cuda:{gpuidx}' if torch.cuda.is_available() else 'cpu'

    sequences, labels = get_sequences_labels(ROOT_DIR=train_dir)

    BATCH_SIZE_LIST = [4, 8, 16, 32, 64]
    LEARNING_RATE_LIST = [1e-2, 1e-4]
    EPOCHS_LIST = [50]
    OPTIMIZER_NAME_LIST = ['sgd', 'adam']
    HIDDEN_SIZE_LIST = [128]

    best_overall_val_acc = 0
    optimal_hyperparams = dict()
    total_params = len(BATCH_SIZE_LIST) * len(LEARNING_RATE_LIST) * len(EPOCHS_LIST) * len(OPTIMIZER_NAME_LIST) * len(
        HIDDEN_SIZE_LIST)

    count = 0
    for batch_size in BATCH_SIZE_LIST:
        for lr in LEARNING_RATE_LIST:
            for epochs in EPOCHS_LIST:
                for op in OPTIMIZER_NAME_LIST:
                    for hidden_size in HIDDEN_SIZE_LIST:
                        count += 1
                        print(f'Grid Training: {count}/{total_params}')
                        if split_data:
                            X = torch.tensor(sequences)
                            y = torch.tensor(labels)

                            dataset = TensorDataset(X, y)

                            train_size = int(TRAIN_VAL_SPLIT * len(dataset))
                            test_size = len(dataset) - train_size
                            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=NUM_WORKERS)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=NUM_WORKERS)

                        else:
                            sequences, labels = get_sequences_labels(ROOT_DIR=train_dir)
                            X = torch.tensor(sequences)
                            y = torch.tensor(labels)
                            train_dataset = TensorDataset(X, y)
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=NUM_WORKERS)

                            sequences, labels = get_sequences_labels(ROOT_DIR=val_dir)
                            X = torch.tensor(sequences)
                            y = torch.tensor(labels)
                            val_dataset = TensorDataset(X, y)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=NUM_WORKERS)

                        num_classes = len(labels_to_gestures)
                        if MODEL_NAME == 'lstm':
                            model = RNN(num_classes, hidden_size=hidden_size)
                        elif MODEL_NAME == 'bilstm':
                            model = BiLSTM(num_classes, hidden_size=hidden_size, bidirectional=BIDIRECTIONAL)
                        else:
                            model = TransformerEncoder(num_classes, hidden_size=hidden_size, device=device)

                        model = model.double().to(device)
                        criterion = nn.CrossEntropyLoss()

                        optimizer = None

                        if op == 'sgd':
                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

                        if op == 'adam':
                            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                               mode='max',
                                                                               factor=0.1,
                                                                               patience=2,
                                                                               verbose=False)
                        best_val_acc = train(model=model,
                                             epochs=epochs,
                                             device=device,
                                             train_loader=train_loader,
                                             val_loader=val_loader,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             )

                        if best_val_acc > best_overall_val_acc:
                            best_overall_val_acc = best_val_acc
                            optimal_hyperparams['batch_size'] = batch_size
                            optimal_hyperparams['lr'] = lr
                            optimal_hyperparams['op'] = op
                            optimal_hyperparams['epochs'] = epochs
                            optimal_hyperparams['hidden_size'] = hidden_size
                            optimal_hyperparams['val_acc'] = best_overall_val_acc

    print('Best found params:')
    print(f'Epochs: {optimal_hyperparams["epochs"]}')
    print(f'Batch Size: {optimal_hyperparams["batch_size"]}')
    print(f'Learning rate: {optimal_hyperparams["lr"]}')
    print(f'Optimizer: {optimal_hyperparams["op"]}')
    print(f'Hidden Size: {optimal_hyperparams["hidden_size"]}')
    print(f'Best validation acc for above params: {optimal_hyperparams["val_acc"]}')


if __name__ == "__main__":
    main()
