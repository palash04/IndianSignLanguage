from collections import defaultdict
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
from ISL_test import test_model
from sklearn.metrics import accuracy_score, classification_report


def train_epoch(model, data_loader, device, criterion, optimizer, scheduler):
    model.train()

    losses = []
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
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
        for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
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


def train(model, epochs, device, train_loader, val_loader, criterion, optimizer, scheduler, model_name):
    history = defaultdict(list)

    best_val_acc = 0

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        print('-' * 10)
        print('Training')
        train_acc, train_loss = train_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        print('\nValidating')
        val_acc, val_loss = val_epoch(model, val_loader, device, criterion)

        print(f'\nTrain Loss: {train_loss}\tTrain Acc: {train_acc}')
        print(f'Val Loss: {val_loss}\tVal Acc: {val_acc}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name}.pth.tar')

    return history


def main():
    parser = argparse.ArgumentParser(description='ISL')
    parser.add_argument('--gpuidx', type=int, default=0, help='GPU index (default: 0)')

    args = parser.parse_args()
    gpuidx = args.gpuidx

    device = f'cuda:{gpuidx}' if torch.cuda.is_available() else 'cpu'

    sequences, labels = get_sequences_labels(ROOT_DIR=train_dir)

    if split_data:
        X = torch.tensor(sequences)
        y = torch.tensor(labels)

        dataset = TensorDataset(X, y)

        train_size = int(TRAIN_VAL_SPLIT * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    else:
        sequences, labels = get_sequences_labels(ROOT_DIR=train_dir)
        X = torch.tensor(sequences)
        y = torch.tensor(labels)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        sequences, labels = get_sequences_labels(ROOT_DIR=val_dir)
        X = torch.tensor(sequences)
        y = torch.tensor(labels)
        val_dataset = TensorDataset(X, y)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(labels_to_gestures)
    if MODEL_NAME == 'lstm':
        model = RNN(num_classes, hidden_size=HIDDEN_SIZE)
    elif MODEL_NAME == 'bilstm':
        model = BiLSTM(num_classes, hidden_size=HIDDEN_SIZE, bidirectional=BIDIRECTIONAL)
    else:
        model = TransformerEncoder(num_classes, hidden_size=HIDDEN_SIZE, device=device)

    model = model.double().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = None

    if OPTIMIZER_NAME == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)

    if OPTIMIZER_NAME == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.1,
                                                           patience=2,
                                                           verbose=False)
    history = train(model=model,
                    epochs=EPOCHS,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_name=f'best_model_{MODEL_NAME}')

    plot_loss_and_acc(history)

    # Check classification report of validation data on best trained model
    model.load_state_dict(torch.load(f'best_model_{MODEL_NAME}.pth.tar'))
    predictions_list, true_list = test_model(model, val_loader, device)

    y_pred = torch.cat(predictions_list).numpy()
    y_true = torch.cat(true_list).numpy()

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Validation dataset accuracy on best model: {acc}')

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=gestures))


if __name__ == "__main__":
    main()
