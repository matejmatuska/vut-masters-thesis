import torch
from sklearn.metrics import f1_score

from dataset import ChronoDataset, Repr1Dataset, SunDataset
from model.baseline import GraphClassifier
from model.chrono import ChronoClassifier
from model.repr1 import Repr1Classifier


def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, f1_average="macro"):
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(data.y.cpu())

            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    f1 = f1_score(all_labels, all_preds, average=f1_average)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, f1, all_preds, all_labels


def get_dataset_factory(which) -> callable:
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


def get_model_factory(which) -> callable:
    if which == "baseline":

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return GraphClassifier(
                edge_dim=dataset[0].edge_attr.size(1),
                port_dim=port_dim,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif which == "chrono":

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return ChronoClassifier(
                input_dim=dataset[0].num_node_features,
                port_dim=port_dim,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif which == "repr1":

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return Repr1Classifier(
                flow_dim=dataset[0]["NetworkFlow"].x.size(1),
                hidden_dim=hidden_dim,
                port_dim=port_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    else:
        # should not reach here, handled by argparse
        raise ValueError("Invalid model type. Choose 'baseline', 'chrono' or 'repr1'.")
    return make_model
