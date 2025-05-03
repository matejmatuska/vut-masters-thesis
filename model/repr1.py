import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    Linear,
    GraphConv,
    global_max_pool,
    global_mean_pool,
)


class Repr1Classifier(torch.nn.Module):

    def __init__(self, input_dim, num_hosts, hidden_dim, num_classes, layers, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        # Host has no features - use embedding
        self.host_embed = torch.nn.Embedding(4000, hidden_dim)

        # data['Host'].x = torch.ones((num_host_nodes, feature_dim))
        #self.lin_flow = Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.convs.append(HeteroConv({
            ('Host', 'communicates', 'NetworkFlow'): GraphConv((hidden_dim, 97), hidden_dim),
            ('NetworkFlow', 'communicates', 'Host'): GraphConv((97, -1), hidden_dim),
            ('NetworkFlow', 'related', 'NetworkFlow'): GraphConv(97, hidden_dim),
        }, aggr='sum'))

        for i in range(layers - 1):
            self.convs.append(HeteroConv({
                ('Host', 'communicates', 'NetworkFlow'): GraphConv(hidden_dim, hidden_dim),
                ('NetworkFlow', 'communicates', 'Host'): GraphConv(hidden_dim, hidden_dim),
                ('NetworkFlow', 'related', 'NetworkFlow'): GraphConv(hidden_dim, hidden_dim),
            }, aggr='sum'))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        # Embed host IDs (assumes they are stored in data['Host'].node_ids)
        x_host = self.host_embed(data['Host'].node_ids)

        #x_host = torch.zeros((data.num_nodes, 1))  # dummy node features
        #x_flow = self.lin_flow(data['NetworkFlow'].x)
        x_dict = {'Host': x_host, 'NetworkFlow': data['NetworkFlow'].x}

        for i in range(len(self.convs) - 1):
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        x_dict = self.convs[-1](x_dict, data.edge_index_dict)

        #pooled_host = global_mean_pool(x_dict['Host'], data['Host'].batch)
        pooled_flow = global_max_pool(x_dict['NetworkFlow'], data['NetworkFlow'].batch)

        #out = self.classifier(torch.cat([pooled_host, pooled_flow], dim=1))
        out = self.classifier(pooled_flow)
        return out
