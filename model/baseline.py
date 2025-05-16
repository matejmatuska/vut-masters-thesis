import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops


class EdgeMPNN(MessagePassing):
    def __init__(self, edge_mlp, dropout):
        super().__init__(aggr="sum")
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean")

        edge_messages = self.edge_mlp(edge_attr)

        prop = self.propagate(edge_index, x=x, edge_attr=edge_messages)
        return prop

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GraphClassifier(torch.nn.Module):  # TODO better names
    def __init__(self, edge_dim, hidden_dim, num_classes, layers, dropout):
        super().__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.gnn_layers = nn.ModuleList(
            [EdgeMPNN(self.edge_mlp, dropout) for _ in range(layers)]
        )
        # self.gnn = EdgeMPNN(edge_dim, self.edge_mlp, dropout=dropout)
        self.pool = global_max_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        # x = self.node_embedding.weight.repeat(data.num_nodes, 1)
        x = torch.zeros((data.num_nodes, 1))  # dummy node features
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, data.edge_attr)
            x = F.relu(x)

        graph_rep = self.pool(x, data.batch)
        return self.classifier(graph_rep)
