from abc import ABC

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

TOTAL_PORTS = 65536  # 2^16
TCP_FLAGS_VALUES = 256  # 8 bit representation


class BaseEdgeMPNN(MessagePassing, ABC):
    """
    Base class for edge message passing neural networks.
    The message function is defined as x_j + edge_attr, where x_j is the neighbor node feature vector.
    """

    def __init__(self, agg, edge_mlp):
        """
        Initializes the edge message passing neural network.

        :param agg: Aggregation method to use, one of ["mean", "max", "sum"].
        :param edge_mlp: MLP to process edge features.
        """
        super().__init__(aggr=agg)
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean")

        edge_messages = self.edge_mlp(edge_attr)

        prop = self.propagate(edge_index, x=x, edge_attr=edge_messages)
        return prop

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class BaseBaselineClassifier(torch.nn.Module):
    """Base class for baseline classifiers."""

    def __init__(
        self, dst_port_embedding, tcp_flags_embedding, gnn_layers, pools, classifier
    ):
        """
        Initializes the baseline classifier.

        :param dst_port_embedding: Embedding layer for destination ports.
        :param tcp_flags_embedding: Embedding layer for TCP flags.
        :param gnn_layers: List of GNN layers.
        :param pools: Pooling method(s) to use one of ["mean", "max", "both"].
        :param classifier: Classifier layer.
        """

        super().__init__()
        self.dst_port_embedding = dst_port_embedding
        self.tcp_flags_embedding = tcp_flags_embedding
        self.gnn_layers = gnn_layers
        self.classifier = classifier

        if pools == "mean":
            self.pools = [global_mean_pool]
        elif pools == "max":
            self.pools = [global_max_pool]
        elif pools == "both":
            self.pools = [global_mean_pool, global_max_pool]
        else:
            raise ValueError(f"Invalid pooling method: {pools}")
        self.npools = len(self.pools)

    def forward(self, data):
        dst_emb = self.dst_port_embedding(data.dst_ports)
        tcp_flags_emb = self.tcp_flags_embedding(data.tcp_flags)

        edge_attr = data.edge_attr
        edge_attr = torch.cat([edge_attr, dst_emb, tcp_flags_emb], dim=1)

        x = torch.zeros((data.num_nodes, 1)).to(data.y.device)  # dummy node features
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr)

        if self.npools == 1:
            pooled = self.pools[0](x, data.batch)
        else:
            max_pool = self.pools[0](x, data.batch)
            mean_pool = self.pools[1](x, data.batch)
            pooled = torch.cat([max_pool, mean_pool], dim=1)
        return self.classifier(pooled)


class BaselineClassifier(BaseBaselineClassifier):
    """Baseline classifier with shared MLP for edge features."""

    class EdgeMPNN(BaseEdgeMPNN):
        def __init__(self, agg, edge_mlp):
            super().__init__(agg=agg, edge_mlp=edge_mlp)

    def __init__(
        self,
        edge_dim,
        hidden_dim,
        port_dim,
        num_classes,
        layers,
        dropout,
        edge_mlp_agg,
        pools,
    ):
        """
        Initializes the baseline classifier.
        :param edge_dim: Dimension of edge features.
        :param hidden_dim: Dimension of hidden layers.
        :param port_dim: Dimension of port features embedding.
        :param num_classes: Output dimension (number of classes).
        :param layers: Number of GNN layers.
        :param dropout: Dropout rate.
        :param edge_mlp_agg: Aggregation method for edge features, one of ["mean", "max"].
        :param pools: Pooling method(s) to use one of ["mean", "max", "both"].
        """
        dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        input_dim = edge_dim + tcp_flags_dim + port_dim
        edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        gnn_layers = nn.ModuleList(
            [self.EdgeMPNN(edge_mlp_agg, edge_mlp) for _ in range(layers)]
        )
        classifier = torch.nn.Sequential(
            torch.nn.Linear(
                2 * hidden_dim if pools == "both" else hidden_dim, hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        super().__init__(
            dst_port_embedding, tcp_flags_embedding, gnn_layers, pools, classifier
        )
