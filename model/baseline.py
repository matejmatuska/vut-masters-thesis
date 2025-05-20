import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

TOTAL_PORTS = 65536  # 2^16
TCP_FLAGS_VALUES = 256  # 8 bit representation


class EdgeMPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__(aggr="sum")
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )


    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean")

        edge_messages = self.edge_mlp(edge_attr)

        prop = self.propagate(edge_index, x=x, edge_attr=edge_messages)
        return prop

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class BaselineClassifier(torch.nn.Module):
    def __init__(self, edge_dim, hidden_dim, port_dim, num_classes, layers, dropout):
        super().__init__()
        tcp_dim = 2

        self.dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        self.tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        input_dim = edge_dim + tcp_dim + port_dim
        # hidden_dim = hidden_dim - port_dim - 2 * tcp_flags_dim

        # input_dim = edge_dim + port_dim
        # hidden_dim = hidden_dim - port_dim
        # input_dim = edge_dim

        self.gnn_layers = nn.ModuleList(
            [EdgeMPNN(input_dim, hidden_dim, dropout) for _ in range(layers)]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        dst_emb = self.dst_port_embedding(data.dst_ports)
        tcp_flags_emb = self.tcp_flags_embedding(data.tcp_flags)
        # tcp_flags_rev_emb = self.tcp_flags_rev_embedding(data.tcp_flags_rev.to(data.y.device))

        # edge_attr = torch.cat([data.edge_attr, dst_emb, tcp_flags_emb, tcp_flags_rev_emb], dim=1)
        edge_attr = data.edge_attr
        edge_attr = torch.cat([edge_attr, dst_emb, tcp_flags_emb], dim=1)


        x = torch.zeros((data.num_nodes, 1)).to(data.y.device)  # dummy node features
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr)

        max_pool = global_max_pool(x, data.batch)
        mean_pool = global_mean_pool(x, data.batch)
        graph_rep = torch.cat([max_pool, mean_pool], dim=1)
        return self.classifier(graph_rep)
