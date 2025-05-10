import os
import sys
import time

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

from model import GraphClassifier

def load_csv(csv):
    try:
        df = pd.read_csv(csv)
    except pd.errors.EmptyDataError:
        print("Empty file")
        return None

    print(df.head())
    # remove the datatype, this could be done when creating the csv
    df.columns = df.columns.str.split().str[1]
    print(df.columns)
    return df


def to_graph(df, draw=False):
    G = nx.DiGraph()
    # Populate the graph with nodes and edges from the DataFrame
    for _, row in df.iterrows():
        src_ip = row['SRC_IP']
        dst_ip = row['DST_IP']
        edge_attr = {
            #'source_port': row['source_port'],
            #'destination_port': row['destination_port'],
            # 'protocol': row['PROTOCOL'],
            'packets': row['PACKETS'],
            'bytes': row['BYTES'],
            'tcp_flags': row['TCP_FLAGS'],
            'tcp_flags_reverse': row['TCP_FLAGS_REV'],
            'dst_port': row['DST_PORT'],
        }

        # Add nodes if they don't exist
        #if src_ip not in G:
        G.add_node(src_ip)
        #if dst_ip not in G:
        G.add_node(dst_ip)

        # Add edge with attributes
        G.add_edge(src_ip, dst_ip, **edge_attr)

    if draw:
        edge_labels = nx.get_edge_attributes(G, 'packets')
        pos = nx.circular_layout(G)

        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
    return G


def make_graph(df, label):
    graph = to_graph(df, draw=False)

    data = from_networkx(graph)
    data.edge_attr = torch.tensor([list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32)
    # TODO add the label to the graph
    data.y = torch.tensor([label], dtype=torch.long)

    print('=== Pytorch Graph ===')

    print(data)
    print(data.edge_attr)
    print(data.y)
    return data


def load_dataset(path, store=False):
    dataset = []

    path = sys.argv[1]
    families = [os.path.join(path, f) for f in os.listdir(path)]

    label_map = {value.split('/')[-1]: index for index, value in enumerate(families, start=0)}

    for family in families:
        file_list = [os.path.join(family, f) for f in os.listdir(family) if f.endswith('.csv')]
        label = label_map[family.split('/')[-1]]
        for f in file_list:
            df = load_csv(f)
            if (df is not None):
                print('=== Processing', f, '===')
                dataset.append(make_graph(df, label))
            else:
                print('=== Skipping', f, '===')

    # TODO what's a good split ratio?
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    if store:
        torch.save(train_dataset, 'train.pt')
        torch.save(test_dataset, 'test.pt')

    return train_dataset, test_dataset, label_map


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        print('out:', out)
        print('labels', data.y)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #print(loss.item())
    return total_loss / (len(train_loader) * 32)


def test(model, test_loader):
    model.eval()

    correct = 0

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            print(pred)
            correct += (pred == data.y).sum().item()

    return correct / (len(test_loader) * 32)


if __name__ == '__main__':
    train_dataset = None
    test_dataset = None

    if len(sys.argv) == 2:
        train_dataset, test_dataset, label_map = load_dataset(sys.argv[1], store=True)
        print('Labels:', len(label_map))
        num_classes = len(label_map)
    elif len(sys.argv) == 3:
        train_dataset = torch.load(sys.argv[1])
        test_dataset = torch.load(sys.argv[2])
        print('Labels:', 2)
        num_classes = 2
    else:
        print("Usage: python prepare_dataset.py <dataset_path>")
        sys.exit(1)

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

    from sklearn.preprocessing import StandardScaler

    # Collect all edge attributes across your dataset
    all_edge_attr = torch.cat([data.edge_attr for data in train_dataset + test_dataset], dim=0)

    # Min-Max normalization using sklearn
    scaler = StandardScaler()
    scaler.fit(all_edge_attr.numpy())  # Fit scaler on all edge attributes

    for data in train_dataset:
        data.edge_attr = torch.tensor(scaler.transform(data.edge_attr.numpy()), dtype=torch.float)
    for data in test_dataset:
        data.edge_attr = torch.tensor(scaler.transform(data.edge_attr.numpy()), dtype=torch.float)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print('=== Pytorch DataLoader ===')
    print(len(train_loader) * train_loader.batch_size)
    print(len(test_loader) * train_loader.batch_size)

    device = torch_geometric.device('cpu')
    #device = torch_geometric.device('auto')
    print(train_dataset[0].edge_attr.size(1))
    model = GraphClassifier(
        edge_dim=train_dataset[0].edge_attr.size(1),
        hidden_dim=100,
        num_classes=num_classes,
    ).to(device)

    learning_rate = 0.001
    # optimizer = torch.optim.Adam([
    #     dict(params=model.conv1.parameters(), weight_decay=5e-4),
    #     dict(params=model.conv2.parameters(), weight_decay=0)
    # ], lr=learning_rate)  # Only perform weight-decay on first convolution.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 200

    init_wandb(
        name=f'Malware (LR={learning_rate}, Hidden={16})',
        lr=learning_rate,
        epochs=epochs,
        hidden_channels=100,
        device=device,
    )

    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train(model, train_loader, optimizer)
        val_acc = test(model, test_loader)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Val=val_acc)
        times.append(time.time() - start)

    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
