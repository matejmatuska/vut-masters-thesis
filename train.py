import argparse
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.transforms import NormalizeFeatures

from dataset import GraphDataset
from model.baseline import GraphClassifier


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split a dataset into training and testing sets.

    :param dataset: The dataset of graphs to split.
    :param train_ratio: The ratio of the dataset to use for training.
    :return: A tuple of (train_dataset, test_dataset).
    """
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_ratio, 1 - train_ratio],
        generator
    )
    return train_dataset, test_dataset


def train(model, train_loader, optimizer, class_weights):
    model.train()

    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, class_weights, last_epoch=False):
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            loss = F.cross_entropy(out, data.y, weight=class_weights)
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(data.y.cpu())

            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    if last_epoch:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()

        conf_matrix = confusion_matrix(all_labels, all_preds)
        log_conf_matrix(conf_matrix, epoch)

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def normalize_data(dataset):
    all_edge_attr = torch.cat([data.edge_attr for data in dataset], dim=0)

    scaler = StandardScaler()
    scaler.fit(all_edge_attr.numpy())  # Fit scaler on all edge attributes

    for data in dataset:
        data.edge_attr = torch.tensor(scaler.transform(data.edge_attr.numpy()), dtype=torch.float)


def compute_class_weights(dataset):
    from sklearn.utils.class_weight import compute_class_weight
    labels = torch.cat([data.y for data in dataset], dim=0).numpy()

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class Weights:", class_weights_tensor)
    return class_weights_tensor


def log_conf_matrix(cm, path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    ticks = np.arange(0, cm.shape[0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ticks, yticklabels=ticks)
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    mlflow.log_figure(plt.gcf(), path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("model", type=str, choices=["baseline"], help="Model to train")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    #parser.add_argument("--hidden", type=int, default=100, help="Hidden dimension size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    #parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "rmsprop"], help="Optimizer to use")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to run training on")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    from dataset import load_dataset_csv, sample_to_graph

    if True:
        dataset = GraphDataset(
            root=args.dataset_path,
            transform=NormalizeFeatures(['edge_attr'])
        )
        num_classes = dataset.num_classes
    else:
        df = load_dataset_csv(args.dataset_path)
        dataset = []
        for sample_name, group in df.groupby('sample'):
            dataset.append(sample_to_graph(group))
            print(f"Processed sample: {sample_name}")

        print("First sample:", dataset[0])
        print("First sample:", dataset[0].edge_attr)

        labels =  df['family'].unique()
        num_classes = len(labels)
        normalize_data(dataset)

    print("First normal sample:", dataset[0])
    class_weights = compute_class_weights(dataset)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8)

    print(f'Number of classes: {num_classes}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('=== Pytorch DataLoader ===')
    print('Train size: ', len(train_dataset))
    print('Test size: ', len(test_dataset))

    device = torch_geometric.device(args.device)

    model = GraphClassifier(
        edge_dim=train_dataset[0].edge_attr.size(1),
        hidden_dim=120,
        num_classes=num_classes,
    ).to(device)

    # optimizer = torch.optim.Adam([
    #     dict(params=model.conv1.parameters(), weight_decay=5e-4),
    #     dict(params=model.conv2.parameters(), weight_decay=0)
    # ], lr=learning_rate)  # Only perform weight-decay on first convolution.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    best_acc = 0
    times = []

    with mlflow.start_run():
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train(model, train_loader, optimizer, class_weights)
            test_loss, acc = test(model, test_loader, class_weights, epoch == args.epochs)

            if acc > best_acc:
                best_acc = acc

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('test_loss', test_loss, step=epoch)
            mlflow.log_metric('accuracy', acc, step=epoch)

            log(Epoch=epoch, Loss=train_loss, Val=test_loss, Acc=acc)
            times.append(time.time() - start)

        log(Best_Acc=best_acc)
        mlflow.log_metric('best_acc', best_acc)
        mlflow.log_metric('median_epoch_time', torch.tensor(times).median())
        print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
