import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, global_max_pool

TOTAL_PORTS = 65536
TCP_FLAGS_VALUES = 256  # 8 bit representation


class Repr2Classifier(torch.nn.Module):
    def __init__(
        self, flow_dim, hidden_dim, port_dim, num_classes, layers, dropout
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        self.tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)
        self.tcp_flags_rev_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        flow_dim = flow_dim + port_dim + 2 * tcp_flags_dim
        self.convs = torch.nn.ModuleList()
        self.convs.append(HeteroConv({
            ('Host', 'communicates', 'NetworkFlow'): GraphConv((hidden_dim, flow_dim), hidden_dim),
            ('NetworkFlow', 'communicates', 'Host'): GraphConv((flow_dim, -1), hidden_dim),
            ('NetworkFlow', 'related', 'NetworkFlow'): GraphConv(flow_dim, hidden_dim),
        }, aggr='sum'))

        for i in range(layers - 1):
            self.convs.append(HeteroConv({
                ('Host', 'communicates', 'NetworkFlow'): GraphConv(hidden_dim, hidden_dim),
                ('NetworkFlow', 'communicates', 'Host'): GraphConv(hidden_dim, hidden_dim),
                ('NetworkFlow', 'related', 'NetworkFlow'): GraphConv(hidden_dim, hidden_dim),
            }, aggr='sum'))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
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
        tcp_flags_rev_emb = self.tcp_flags_rev_embedding(
            data["NetworkFlow"].tcp_flags_rev.to(data.y.device)
        )

        # dummy node features
        x_host = torch.zeros((data.num_nodes, self.hidden_dim)).to(data.y.device)
        x_flow = torch.cat(
            [data["NetworkFlow"].x, dst_emb, tcp_flags_emb, tcp_flags_rev_emb], dim=1
        )
        x_dict = {'Host': x_host, 'NetworkFlow': x_flow}

        for i in range(len(self.convs) - 1):
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        x_dict = self.convs[-1](x_dict, data.edge_index_dict)

        # pooled_host = global_mean_pool(x_dict['Host'], data['Host'].batch)
        pooled_flow = global_max_pool(x_dict['NetworkFlow'], data['NetworkFlow'].batch)

        #out = self.classifier(torch.cat([pooled_host, pooled_flow], dim=1))
        out = self.classifier(pooled_flow)
        return out
