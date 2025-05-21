import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphConv, global_max_pool

TOTAL_PORTS = 65536  # 2^16
TCP_FLAGS_VALUES = 256  # 8 bit representation


class Repr1Classifier(torch.nn.Module):
    """A classifier for the graph representation 1"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        port_dim,
        num_classes,
        layers,
        dropout,
        layer_type,
        **kwargs,
    ):
        """
        Initializes the Repr1Classifier.

        :param input_dim: Dimension of input features.
        :param hidden_dim: Dimension of hidden layers.
        :param port_dim: Dimension of port features embedding.
        :param num_classes: Output dimension (number of classes).
        :param layers: Number of GNN layers.
        :param dropout: Dropout rate.
        :param layer_type: Type of GNN layer to use. Can be 'graphconv', 'gcn', or 'gat'.
        :param kwargs: Number of heads for GAT layer, if applicable.
        """
        super().__init__()
        self.dropout = dropout

        self.dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        self.tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        input_dim = input_dim + tcp_flags_dim + port_dim

        self.convs = torch.nn.ModuleList()
        if layer_type in ("graphconv", "gcn"):
            if layer_type == "graphconv":
                _layer_type = GraphConv
            elif layer_type == "gcn":
                _layer_type = GCNConv

            self.convs.append(_layer_type(input_dim, hidden_dim))
            for _ in range(layers - 2):
                self.convs.append(_layer_type(hidden_dim, hidden_dim))
            self.convs.append(_layer_type(hidden_dim, hidden_dim))
        elif layer_type == "gat":
            heads = kwargs.get("heads", 2)
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
            for _ in range(layers - 2):
                self.convs.append(
                    GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
                )
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
            )
        else:
            raise ValueError(
                f"Invalid layer type: {layer_type}. Choose from ['graphconv', 'gcn', 'gat']"
            )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        dst_emb = self.dst_port_embedding(data.dst_ports.to(data.x.device))
        tcp_flags_emb = self.tcp_flags_embedding(data.tcp_flags.to(data.x.device))

        x = torch.cat([data.x.float(), dst_emb, tcp_flags_emb], dim=1)

        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_max_pool(x, data.batch)
        x = self.fc(x)
        return x
