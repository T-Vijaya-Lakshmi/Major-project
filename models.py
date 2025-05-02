import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from layers import GCN, HGPSLPool

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features  # Input feature size
        self.nhid = args.nhid  # Hidden layer size (128)
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        # Define GCN layers
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)

        # Define pooling layers
        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio)
        
        self.lin_expand = torch.nn.Linear(self.nhid, self.nhid * 2)  # (128 → 256)
        
        # FIX: Adjusting feature sizes to avoid shape mismatch
        self.lin_pool = torch.nn.Linear(self.nhid * 2, self.nhid)  # Ensure consistent feature size

        # FIX: Ensure `lin1` input size is correct
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)  # Now expects 128 → 128
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)  # Output layer

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x, edge_index)  # (num_nodes, nhid) -> (147, 128)
        x2 = self.conv2(x1, edge_index)  # (147, 128)

        num_graphs = batch.max().item() + 1  # Get number of graphs in batch

        # Global pooling
        pooled1 = gmp(x2, batch)  # (num_graphs, nhid) -> (4, 128)
        pooled2 = gap(x2, batch)  # (num_graphs, nhid) -> (4, 128)

        # Concatenate pooled features
        x3 = torch.cat([pooled1, pooled2], dim=1)  # (4, 256)
        #x3 = self.lin_pool(x3)  # (4, 128) - Reshaped properly

        # FIX: Expand `x3` to match node-level feature size
        x3 = x3[batch]  # Expands (4, 128) → (147, 128)
        
         #  Apply expansion to x1 and x2 to match x3
        x1 = self.lin_expand(x1)  # (147, 128) → (147, 256)
        x2 = self.lin_expand(x2)  # (147, 128) → (147, 256)
        #  Debugging: Print shapes before adding
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}, x3 shape: {x3.shape}")

        # Ensure tensor sizes match before addition
        assert x1.shape == x2.shape == x3.shape, f"Shape mismatch: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}"

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)  # No shape mismatch

        x = F.relu(self.lin1(x))  # (147, 128) Now expects `128 → 128`
         #  Final Graph-Level Pooling (Fixes Batch Size Mismatch)
        x = gmp(x, batch)  # (num_graphs, 128) 

        x = self.lin2(x)  # (147, num_classes)

        return F.log_softmax(x, dim=-1)
