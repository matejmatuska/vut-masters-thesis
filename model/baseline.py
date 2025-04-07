import numpy as np
import networkx as nx

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, MessagePassing, NNConv
from torch_geometric.utils import add_self_loops
from collections import defaultdict


class EdgeMPNN(MessagePassing):
    def __init__(self,  edge_mlp, dropout=0):
        super().__init__(aggr='sum')
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean')

        edge_messages = self.edge_mlp(edge_attr)

        prop = self.propagate(edge_index, x=x, edge_attr=edge_messages)
        return prop

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GraphClassifier(torch.nn.Module):
    def __init__(self, edge_dim, hidden_dim, num_classes, dropout=0):
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
        self.gnn_layers = nn.ModuleList([EdgeMPNN(self.edge_mlp, dropout) for _ in range(1)])
        #self.gnn = EdgeMPNN(edge_dim, self.edge_mlp, dropout=dropout)
        self.pool = global_max_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        #x = self.node_embedding.weight.repeat(data.num_nodes, 1)
        x = torch.zeros((data.num_nodes, 1))  # dummy node features
        #x = self.gnn(x, data.edge_index, data.edge_attr)
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, data.edge_attr)
        graph_rep = self.pool(x, data.batch)
        return self.classifier(graph_rep)


def row_to_graph(
    df,
    draw=False,
    attributes=[
        'PACKETS',
        'PACKETS_REV',
        'BYTES',
        'BYTES_REV',
        # TODO 'TCP_FLAGS',
        # TODO 'TCP_FLAGS_REV',
        'PROTOCOL',
        'SRC_PORT',
        'DST_PORT',
    ],
):
    """
    Convert a row of dataset DataFrame to a digraph.
    """
    def pad_ppi(series, to_len=30, value=0):
        return np.pad(series, (0, to_len - len(series)), 'constant', constant_values=value)

    G = nx.MultiDiGraph()
    # populate the graph with nodes and edges from the DataFrame
    for _, row in df.iterrows():
        src_ip = row['SRC_IP']
        dst_ip = row['DST_IP']
        edge_attr = {attr: row[attr] for attr in attributes}

        # TODO how long?
        edge_attr['PPI_PKT_LENGTHS'] = pad_ppi(row['PPI_PKT_LENGTHS'], value=0)
        edge_attr['PPI_PKT_DIRECTIONS'] = pad_ppi(row['PPI_PKT_DIRECTIONS'], value=0)
        edge_attr['PPI_PKT_TIMES'] = pad_ppi(row['PPI_PKT_TIMES'], value=0)

        G.add_edge(src_ip, dst_ip, **edge_attr)

    # if draw:
    #     edge_labels = nx.get_edge_attributes(G, 'packets')
    #     pos = nx.circular_layout(G)

    #     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #     plt.show()


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

