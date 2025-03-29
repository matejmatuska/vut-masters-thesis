import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing, NNConv
from torch_geometric.utils import add_self_loops


class EdgeMPNN(MessagePassing):
    def __init__(self, edge_dim, hidden_dim, dropout=0):
        super().__init__(aggr='max')
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0)

        edge_messages = self.edge_mlp(edge_attr)

        prop = self.propagate(edge_index, x=x, edge_weight=edge_messages)
        return prop

    def message(self, edge_weight):
        return edge_weight  # Use transformed edge features as messages


class GraphClassifier(torch.nn.Module):
    def __init__(self, edge_dim, hidden_dim, num_classes, dropout=0):
        super().__init__()
        self.gnn = EdgeMPNN(edge_dim, hidden_dim)
        self.pool = global_mean_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        #x = self.node_embedding.weight.repeat(data.num_nodes, 1)
        x = torch.zeros((data.num_nodes, 1))  # dummy node features
        x = self.gnn(x, data.edge_index, data.edge_attr)
        graph_rep = self.pool(x, data.batch)
        return self.classifier(graph_rep)


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        """
        GCN for graph classification using edge features.

        Args:
            input_dim (int): Initial input dimension for node features (dummy if x is None).
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Number of hidden units in GCN layers.
            output_dim (int): Number of output classes (graph labels).
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
        """
        super(GCNGraphClassifier, self).__init__()

        self.node_embedding = torch.nn.Embedding(1, input_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * hidden_dim)  # input_dim * output_dim
        )

        # Define the NNConv layer
        self.conv1 = NNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp,
            aggr='add'  # Aggregation method ('mean', 'add', 'max')
        )
        # Output layer
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        # Learnable node embeddings (if no node features are present)
        # self.node_embedding = torch.nn.Embedding(1, input_dim)
        #
        # # Define GCN layers
        # self.convs = torch.nn.ModuleList()
        # self.convs.append(GCNConv(input_dim, hidden_dim))
        # for _ in range(num_layers - 1):
        #     self.convs.append(GCNConv(hidden_dim, hidden_dim))
        #
        # # Fully connected layer for classification
        # self.fc = torch.nn.Linear(hidden_dim, output_dim)
        #
        # # MLP to process edge features
        # self.edge_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(edge_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, hidden_dim)
        # )
        # self.dropout = dropout


    def forward(self, data):
        """
        Forward pass for the GCN model.

        Args:
            data: A PyTorch Geometric Data object with edge_attr, edge_index, batch.

        Returns:
            logits (torch.Tensor): Predicted logits for each graph in the batch.
        """
        edge_attr, edge_index, batch = data.edge_attr, data.edge_index, data.batch

        # print("PRINT", edge_attr, edge_index, batch)
        if data.x is None:
            x = self.node_embedding.weight.repeat(data.num_nodes, 1)
            #print('HERE')
        else:
            x = data.x
            print('!!! data.x is NOT None')

         # Message passing
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        # Final classification
        return self.linear(x)


        # #print("PRINT", edge_attr, edge_index, batch)
        #
        # # Initialize dummy node features if x is None
        # if data.x is None:
        #     x = self.node_embedding.weight.repeat(data.num_nodes, 1)
        #     print('HERE')
        # else:
        #     x = data.x
        #
        # # Process edge features through the MLP
        # edge_features = self.edge_mlp(edge_attr)
        #
        # # Apply GCN layers
        # for conv in self.convs:
        #     x = conv(x, edge_index, edge_weight=edge_features.sum(dim=1))  # Use edge features as weights
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #
        # # Global pooling to get graph-level embeddings
        # x = global_mean_pool(x, batch)
        #
        # # Fully connected layer
        # x = self.fc(x)
        # return x
