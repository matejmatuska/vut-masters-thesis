import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_max_pool

TOTAL_PORTS = 65536  # 2^16
TCP_FLAGS_VALUES = 256  # 8 bit representation


class Repr1Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, port_dim, num_classes, layers, dropout):
        super().__init__()
        self.dropout = dropout

        self.dst_port_embedding = torch.nn.Embedding(TOTAL_PORTS, port_dim)
        tcp_flags_dim = 2
        self.tcp_flags_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)
        # self.tcp_flags_rev_embedding = torch.nn.Embedding(TCP_FLAGS_VALUES, tcp_flags_dim)

        input_dim = input_dim + tcp_flags_dim + port_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(input_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.convs.append(GraphConv(hidden_dim, hidden_dim))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        dst_emb = self.dst_port_embedding(data.dst_ports.to(data.x.device))
        tcp_flags_emb = self.tcp_flags_embedding(data.tcp_flags.to(data.x.device))
        # tcp_flags_rev_emb = self.tcp_flags_rev_embedding(data.tcp_flags_rev.to(data.x.device))

        # x = torch.cat([data.x.float(), dst_emb, tcp_flags_emb, tcp_flags_rev_emb], dim=1)
        x = torch.cat([data.x.float(), dst_emb, tcp_flags_emb], dim=1)

        # x = data.x.float()
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_max_pool(x, data.batch)
        x =  self.fc(x)
        return x
