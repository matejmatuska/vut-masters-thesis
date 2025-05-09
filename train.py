import argparse
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.transforms import NormalizeFeatures

from dataset import ChronoDataset, Repr1Dataset, SunDataset
from model.baseline import GraphClassifier
from model.chrono import ChronoClassifier
from model.repr1 import Repr1Classifier


def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, device, last_epoch=False):
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(data.y.cpu())

            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    if last_epoch:
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        log_conf_matrix(conf_matrix, epoch)

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def compute_class_weights(dataset):
    labels = torch.cat([data.y for data in dataset], dim=0).numpy()

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
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
    parser.add_argument("model", type=str, choices=["baseline", "chrono", "repr1"], help="Model to train")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    #parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "rmsprop"], help="Optimizer to use")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to run training on")
    return parser.parse_args()


def get_dataset_factory(which) -> (callable):
    if which == "baseline":
        def creator(root, split, **kwargs) -> SunDataset:
            return SunDataset(
                root,
                split,
                **kwargs,
            )
    elif which == "chrono":
        def creator(root, split, **kwargs) -> ChronoDataset:
            return ChronoDataset(
                root,
                split,
                **kwargs,
            )
    elif which == "repr1":
        def creator(root, split, **kwargs) -> Repr1Dataset:
            return Repr1Dataset(
                root,
                split,
                **kwargs,
            )
    else:
        raise ValueError("Invalid model type. Choose 'baseline', 'chrono' or 'repr1'.")
    return creator


def get_model_factory(which) -> (callable):
    if which == "baseline":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return GraphClassifier(
                edge_dim=dataset[0].edge_attr.size(1),
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif which == "chrono":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return ChronoClassifier(
                input_dim=dataset[0].num_node_features,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif args.model == "repr1":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return Repr1Classifier(
                input_dim=97,
                num_hosts=0, # TODO unused
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    else:
        # should not reach here, handled by argparse
        raise ValueError("Invalid model type. Choose 'baseline', 'chrono' or 'repr1'.")
    return make_model


if __name__ == '__main__':
    args = parse_args()
    print("Model:", args.model)

    dset_factory = get_dataset_factory(args.model)
    transform = NormalizeFeatures()
    train_set = dset_factory(args.dataset_path, "train", transform=transform)
    val_set = dset_factory(args.dataset_path, "val", transform=transform)
    test_set = dset_factory(args.dataset_path, "test", transform=transform)

    print(
        f"Dataset samples: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
    )
    print("First train sample:", train_set[0])
    print("First train sample:", train_set[1])
    print("First train sample:", train_set[2])
    print("First train sample:", train_set[3])
    print("First train sample:", train_set[4])

    num_classes = train_set.num_classes
    device = torch.device(args.device)

    class_weights = compute_class_weights(train_set).to(device)

    print(f'Number of classes: {num_classes}')
    print("Class Weights:", class_weights)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    model_factory = get_model_factory(args.model)
    model = model_factory(
        train_set,
        hidden_dim=args.hidden,
        dropout=args.dropout,
        nlayers=args.layers,
    ).to(device)

    # optimizer = torch.optim.Adam([
    #     dict(params=model.conv1.parameters(), weight_decay=5e-4),
    #     dict(params=model.conv2.parameters(), weight_decay=0)
    # ], lr=learning_rate)  # Only perform weight-decay on first convolution.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("xmatus36-gnns")
    with mlflow.start_run():
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_params(vars(args))

        best_acc = 0
        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, device)

            test_loss, acc = validate(model, val_loader, criterion, device, epoch == args.epochs)

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
