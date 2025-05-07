import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class ChronoClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, layers, dropout=0.0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim // 2))
        # self.convs.append(GCNConv(hidden_dim // 2, hidden_dim // 4))
        self.convs.append(GCNConv(hidden_dim // 2, hidden_dim // 4))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            #torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_max_pool(x, batch)
        x =  self.fc(x)
        return F.log_softmax(x, dim=-1)
