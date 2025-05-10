import argparse
import os
import sys
import time

from ast import literal_eval
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

import torch
import torch.nn.functional as F

import torch_geometric
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.utils import from_networkx

from sklearn.preprocessing import StandardScaler

from model import GraphClassifier


def row_to_graph(
    df,
    draw=False,
    attributes=['PACKETS', 'BYTES', 'TCP_FLAGS', 'TCP_FLAGS_REV', 'DST_PORT', 'PROTOCOL', 'SRC_PORT'],
):
    """
    Convert a row of dataset DataFrame to a digraph.
    """
    def pad_ppi(ppi_str, to_len=30, value=0):
        return np.pad(ppi_str, (0, to_len - len(ppi_str)), 'constant', constant_values=value)

    G = nx.MultiDiGraph()
    # Populate the graph with nodes and edges from the DataFrame
    for _, row in df.iterrows():
        src_ip = row['SRC_IP']
        dst_ip = row['DST_IP']
        edge_attr = {attr: row[attr] for attr in attributes}

        edge_attr['PPI_PKT_LENGTHS'] = pad_ppi(row['PPI_PKT_LENGTHS'], value=-1)
        #edge_attr['PPI_PKT_TIMES'] = pad_ppi(row['PPI_PKT_TIMES'], value=0)
        #print(edge_attr['PPI_PKT_TIMES'])

        # Add edge with attributes
        G.add_edge(src_ip, dst_ip, **edge_attr)

    if draw:
        edge_labels = nx.get_edge_attributes(G, 'packets')
        pos = nx.circular_layout(G)

        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()


    def aggregate_edges(graph, aggfunc=np.mean) -> nx.DiGraph:
        """
        Aggregate edges of a multi-digraph to a standard di-graph.
        """
        ret = nx.DiGraph()
        edge_data = defaultdict(lambda: defaultdict(list))

        for u, v, data in graph.edges(data=True):
            for key, val in data.items():
                edge_data[(u, v)][key].append(val)

        for (u, v), data in edge_data.items():
            agg_data = {key: aggfunc(val_list) for key, val_list in data.items()}
            ret.add_edge(u, v, **agg_data)

        return ret

    agg_G = aggregate_edges(G)
    assert not any(nx.isolates(agg_G))
    return agg_G


def sample_to_graph(df):
    graph = row_to_graph(df, draw=False)
    label = df['label_encoded'].iloc[0]

    # make torch data
    data = from_networkx(graph)
    data.edge_attr = torch.tensor([list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32)
    # TODO add the label to the graph
    data.y = torch.tensor([label], dtype=torch.long)


    return data


def load_dataset_csv(path, store=False):
    """
    Load a dataset from a CSV file.
    """
    dataset = []
    df = pd.read_csv(path)
    df = df[~df['family'].isin(['LOKIBOT', 'XWORM', 'NETWIRE', 'SLIVER', 'AGENTTESLA', 'WARZONERAT', 'COBALTSTRIKE'])]

    min_samples = 150
    max_samples = 300

    sample_counts = (
        df[['family', 'sample']]
            .drop_duplicates()  # Get unique family-sample combinations
            .groupby('family')  # Group by family
            .size()  # Count the number of unique samples per family
    )

    # filter families that meet the minimum sample requirement
    valid_families = sample_counts[sample_counts >= min_samples].index

    # filter the original DataFrame to keep only valid families
    df_filtered = df[df['family'].isin(valid_families)]
    print(df_filtered)

    # reduce samples per family to the max limit
    selected_samples = (
        df_filtered[['family', 'sample']].drop_duplicates()
            .groupby('family', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), max_samples)))
    )

    # merge to keep only flows from selected samples
    df = df_filtered.merge(selected_samples, on=['family', 'sample'])
    print(df[['family', 'sample']].drop_duplicates()['family'].value_counts())
    print(df[['family', 'sample']].drop_duplicates()['family'].value_counts().sum())


    df['label_encoded'], _ = pd.factorize(df['family'])
    print(f'Encoded familites: {df.groupby("family")["label_encoded"].first()}')

    df['PPI_PKT_LENGTHS'] = df['PPI_PKT_LENGTHS'].str.replace('|', ',')
    df['PPI_PKT_LENGTHS'] = df['PPI_PKT_LENGTHS'].apply(literal_eval)

    for sample_name, group in df.groupby('sample'):
        dataset.append(sample_to_graph(group))


    # TODO what's a good split ratio?
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    if store:
        torch.save(train_dataset, 'train.pt')
        torch.save(test_dataset, 'test.pt')

    return train_dataset, test_dataset, df['family'].unique()



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


def test(model, test_loader, class_weights):
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

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def normalize_data(train_dataset, test_dataset):
    all_edge_attr = torch.cat([data.edge_attr for data in train_dataset + test_dataset], dim=0)

    scaler = StandardScaler()
    scaler.fit(all_edge_attr.numpy())  # Fit scaler on all edge attributes

    for data in train_dataset:
        data.edge_attr = torch.tensor(scaler.transform(data.edge_attr.numpy()), dtype=torch.float)
    for data in test_dataset:
        data.edge_attr = torch.tensor(scaler.transform(data.edge_attr.numpy()), dtype=torch.float)


def compute_class_weights(dataset):
    from sklearn.utils.class_weight import compute_class_weight
    labels = torch.cat([data.y for data in dataset], dim=0).numpy()

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class Weights:", class_weights_tensor)
    return class_weights_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    # Positional argument (unnamed) for dataset path
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")

    # Common training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="Learning rate for optimizer")
    #parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizers like SGD")
    #parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "rmsprop"], help="Optimizer to use")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to run training on")
    # parser.add_argument("--save_model", type=str, default=None, help="Path to save the trained model")
    # parser.add_argument("--load_model", type=str, default=None, help="Path to load a pre-trained model")

    return parser.parse_args()


if __name__ == '__main__':

    train_dataset = None
    test_dataset = None

    args = parse_args()

    #if len(sys.argv) == 2:
    #    train_dataset, test_dataset, labels = load_dataset_csv(sys.argv[1], store=True)
    #    print('Labels:', len(labels))
    #    num_classes = len(labels)
    #elif len(sys.argv) == 3:
    #    train_dataset = torch.load(sys.argv[1])
    #    test_dataset = torch.load(sys.argv[2])
    #    print('Labels:', 2)
    #    num_classes = 2
    #else:
    #    print("Usage: python prepare_dataset.py <dataset_path>")
    #    sys.exit(1)

    train_dataset, test_dataset, labels = load_dataset_csv(args.dataset_path, store=True)
    print('Labels:', len(labels))
    num_classes = len(labels)
    OUTPUT_DIM = num_classes

    # for data in train_dataset:
    #     edge_attr = data.edge_attr
    #     mean = edge_attr.mean(dim=0, keepdim=True)
    #     std = edge_attr.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
    #     data.edge_attr = (edge_attr - mean) / std
    #
    # for data in test_dataset:
    #     edge_attr = data.edge_attr
    #     mean = edge_attr.mean(dim=0, keepdim=True)
    #     std = edge_attr.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
    #     data.edge_attr = (edge_attr - mean) / std


    # Collect all edge attributes across your dataset
    normalize_data(train_dataset, test_dataset)

    class_weights = compute_class_weights(train_dataset + test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('=== Pytorch DataLoader ===')
    print('Train size: ', len(train_loader) * train_loader.batch_size)
    print('Test size: ', len(test_loader) * train_loader.batch_size)

    device = torch_geometric.device(args.device)
    print(train_dataset[0].edge_attr.size(1))

    model = GraphClassifier(
        edge_dim=train_dataset[0].edge_attr.size(1),
        hidden_dim=100,
        num_classes=num_classes,
    ).to(device)

    # optimizer = torch.optim.Adam([
    #     dict(params=model.conv1.parameters(), weight_decay=5e-4),
    #     dict(params=model.conv2.parameters(), weight_decay=0)
    # ], lr=learning_rate)  # Only perform weight-decay on first convolution.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    init_wandb(
        name=f'Malware (LR={args.learning_rate}, Hidden={args.batch_size})',
        lr=args.learning_rate,
        epochs=args.epochs,
        hidden_channels=100,
        device=device,
    )

    best_acc = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, train_loader, optimizer, class_weights)
        test_loss, acc = test(model, test_loader, class_weights)

        if acc > best_acc:
            best_acc = acc

        log(Epoch=epoch, Loss=train_loss, Val=test_loss, Acc=acc)
        times.append(time.time() - start)

    log(Best_Acc=best_acc)
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
