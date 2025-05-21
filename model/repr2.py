import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, HeteroConv, SAGEConv, global_max_pool

TOTAL_PORTS = 65536  # 2^16
TCP_FLAGS_VALUES = 256  # 8 bit representation


class Repr2Classifier(torch.nn.Module):
    """A classifier for the graph representation 2."""

    def __init__(
        self,
        flow_dim,
        hidden_dim,
        port_dim,
        num_classes,
        layers,
        dropout,
        layer_type,
        **kwargs,
    ):
        """
        Initializes the Repr2Classifier.

        :param flow_dim: Dimension of flow features.
        :param hidden_dim: Dimension of hidden layers.
        :param port_dim: Dimension of port features embedding.
        :param num_classes: Output dimension (number of classes).
        :param layers: Number of GNN layers.
        :param dropout: Dropout rate.
        :param layer_type: Type of GNN layer to use. Can be 'graphconv', 'sage', or 'gat'.
        :param kwargs: Number of heads for GAT layer, if applicable.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        port_dim = 1

        self.dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        self.tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        flow_dim = flow_dim + tcp_flags_dim + port_dim

        self.convs = torch.nn.ModuleList()
        if layer_type in ("graphconv", "sage"):
            if layer_type == "graphconv":
                _layer_type = GraphConv
            elif layer_type == "sage":
                _layer_type = SAGEConv

            self.convs.append(
                HeteroConv(
                    {
                        ("Host", "communicates", "NetworkFlow"): _layer_type(
                            (1, flow_dim), hidden_dim
                        ),
                        ("NetworkFlow", "communicates", "Host"): _layer_type(
                            (flow_dim, 1), hidden_dim
                        ),
                        ("NetworkFlow", "related", "NetworkFlow"): _layer_type(
                            flow_dim, hidden_dim
                        ),
                    },
                    aggr="sum",
                )
            )
            for i in range(layers - 1):
                self.convs.append(
                    HeteroConv(
                        {
                            ("Host", "communicates", "NetworkFlow"): _layer_type(
                                hidden_dim, hidden_dim
                            ),
                            ("NetworkFlow", "communicates", "Host"): _layer_type(
                                hidden_dim, hidden_dim
                            ),
                            ("NetworkFlow", "related", "NetworkFlow"): _layer_type(
                                hidden_dim, hidden_dim
                            ),
                        },
                        aggr="sum",
                    )
                )

        elif layer_type == "gat":
            heads = kwargs.get("heads", 4)
            self.convs.append(
                HeteroConv(
                    {
                        ("Host", "communicates", "NetworkFlow"): GATConv(
                            (1, flow_dim),
                            hidden_dim,
                            heads=heads,
                            concat=True,
                            add_self_loops=False,
                        ),
                        ("NetworkFlow", "communicates", "Host"): GATConv(
                            (flow_dim, 1),
                            hidden_dim,
                            heads=heads,
                            concat=True,
                            add_self_loops=False,
                        ),
                        ("NetworkFlow", "related", "NetworkFlow"): GATConv(
                            flow_dim,
                            hidden_dim,
                            heads=heads,
                            concat=True,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )
            for i in range(layers - 1):
                self.convs.append(
                    HeteroConv(
                        {
                            ("Host", "communicates", "NetworkFlow"): GATConv(
                                hidden_dim * heads,
                                hidden_dim,
                                heads=heads,
                                concat=True,
                                add_self_loops=False,
                            ),
                            ("NetworkFlow", "communicates", "Host"): GATConv(
                                hidden_dim * heads,
                                hidden_dim,
                                heads=heads,
                                concat=True,
                                add_self_loops=False,
                            ),
                            ("NetworkFlow", "related", "NetworkFlow"): GATConv(
                                hidden_dim * heads,
                                hidden_dim,
                                heads=heads,
                                concat=True,
                                add_self_loops=False,
                            ),
                        },
                        aggr="sum",
                    )
                )
            self.convs.append(
                HeteroConv(
                    {
                        ("Host", "communicates", "NetworkFlow"): GATConv(
                            hidden_dim * heads,
                            hidden_dim * heads // 2,
                            heads=heads // 2,
                            concat=False,
                            add_self_loops=False,
                        ),
                        ("NetworkFlow", "communicates", "Host"): GATConv(
                            hidden_dim * heads,
                            hidden_dim * heads // 2,
                            heads=heads // 2,
                            concat=False,
                            add_self_loops=False,
                        ),
                        ("NetworkFlow", "related", "NetworkFlow"): GATConv(
                            hidden_dim * heads,
                            hidden_dim * heads // 2,
                            heads=heads // 2,
                            concat=False,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )

        else:
            raise ValueError(
                f"Invalid layer type: {layer_type}. Choose from ['graphconv', 'gcn', 'gat']"
            )

        classifier_input_dim = (
            hidden_dim * kwargs.get("heads", 1) * 2
            if layer_type == "gat"
            else hidden_dim
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                classifier_input_dim, hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        dst_emb = self.dst_port_embedding(
            data["NetworkFlow"].dst_ports.to(data.y.device)
        )
        tcp_flags_emb = self.tcp_flags_embedding(
            data["NetworkFlow"].tcp_flags.to(data.y.device)
        )

        # dummy node features
        x_host = torch.zeros((data["Host"].num_nodes, 1)).to(data.y.device)
        x_flow = data["NetworkFlow"].x
        x_flow = torch.cat([x_flow, dst_emb, tcp_flags_emb], dim=1)

        x_dict = {"Host": x_host, "NetworkFlow": x_flow}

        for i in range(len(self.convs) - 1):
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        x_dict = self.convs[-1](x_dict, data.edge_index_dict)

        pooled_flow = global_max_pool(x_dict["NetworkFlow"], data["NetworkFlow"].batch)

        # out = self.classifier(torch.cat([pooled_host, pooled_flow], dim=1))
        out = self.classifier(pooled_flow)
        return out
