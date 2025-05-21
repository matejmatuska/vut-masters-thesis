from itertools import product

import torch
from sklearn.metrics import f1_score

from dataset import BaselineDataset, Repr1Dataset, Repr2Dataset
from model.baseline import BaselineClassifier
from model.repr1 import Repr1Classifier
from model.repr2 import Repr2Classifier


def get_allowed_models():
    def make_model_combinations(*args):
        return ["-".join(x) for x in list(product(*args))]

    baseline_models = make_model_combinations(
        ("baseline",),
        ("mean", "max", "sum"),
        ("mean", "max", "both"),
    )
    repr1_models = make_model_combinations(
        ("repr1",),
        ("graphconv", "gcn", "gat"),
    )
    repr2_models = make_model_combinations(
        ("repr2",),
        ("graphconv", "sage", "gat"),
    )

    return baseline_models + repr1_models + repr2_models


def train(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    :param model: The model to train.
    :param loader: The data loader for the training set.
    :param optimizer: The optimizer to use.
    :param criterion: The loss function to use.
    :param device: The device to use (CPU or GPU).
    :return: The average loss for the epoch.
    """
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
    """
    Evaluate the model on the validation or test set.

    :param model: The model to evaluate.
    :param loader: The data loader for the validation or test set.
    :param criterion: The loss function to use.
    :param device: The device to use (CPU or GPU).
    :param f1_average: The type of F1 score to compute. Can be 'macro', 'micro' or 'weighted'.
    :return: The average loss, accuracy, and F1 score, and the predictions and labels.
    """
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
    """
    Returns a function that creates a dataset of the specified type.

    :param which: The type of dataset to create. Can be 'baseline', 'repr1' or 'repr2'.
    :return: A function that creates a dataset of the specified type.
    """
    if which == "baseline":

        def creator(root, split, **kwargs) -> BaselineDataset:
            return BaselineDataset(
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
    elif which == "repr2":

        def creator(root, split, **kwargs) -> Repr2Dataset:
            return Repr2Dataset(
                root,
                split,
                **kwargs,
            )
    else:
        raise ValueError("Invalid model type. Choose 'baseline', 'repr1' or 'repr2'.")
    return creator


def get_model_factory(which) -> callable:
    """
    Returns a function that creates a model of the specified type.

    :param which: The type of model to create. Can be 'baseline', 'repr1' or 'repr2'.
    :return: A function that creates a model of the specified type.
    """
    if which[0] == "baseline":
        assert len(which) == 3
        edge_mlp_agg = which[1]
        pools = which[2]

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return BaselineClassifier(
                edge_dim=dataset[0].edge_attr.size(1),
                edge_mlp_agg=edge_mlp_agg,
                port_dim=port_dim,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
                pools=pools,
            )

    elif which[0] == "repr1":
        assert len(which) == 2
        layer_type = which[1]

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return Repr1Classifier(
                input_dim=dataset[0].num_node_features,
                port_dim=port_dim,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
                layer_type=layer_type,
            )
    elif which[0] == "repr2":
        assert len(which) == 2
        layer_type = which[1]

        def make_model(dataset, hidden_dim, port_dim, dropout, nlayers):
            return Repr2Classifier(
                flow_dim=dataset[0]["NetworkFlow"].x.size(1),
                hidden_dim=hidden_dim,
                port_dim=port_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
                layer_type=layer_type,
            )
    else:
        # should not reach here, handled by argparse
        raise ValueError("Invalid model type. Choose 'baseline', 'repr1' or 'repr2'.")
    return make_model
