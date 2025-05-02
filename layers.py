import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, coalesce
from torch_scatter import scatter_add
from torch_sparse import spspmm

# Custom Top-K function
def custom_topk(x, ratio):
    num_nodes = x.size(0)
    
    # Ensure x doesn't have NaN or Inf values
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(" Warning: `x` contains NaN or Inf values! Replacing with zeros.")
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    # Ensure k is within valid range
    k = min(max(1, int(ratio * num_nodes)), num_nodes)

    # Perform top-k selection
    _, indices = torch.topk(x.view(-1), k, largest=True, sorted=False)
    
    return indices
# Edge filtering function
def filter_adj(edge_index, edge_attr, mask):
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr

class TwoHopNeighborhood:
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes
        fill = 1e16

        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)
        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, 1).expand(-1, edge_attr.size(1))
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op="min")
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

# GCN Layer
class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(aggr='add')
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out

# HGPSLPool Layer
class HGPSLPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8):
        super(HGPSLPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.att = Parameter(torch.Tensor(1, in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = GCN(in_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        perm = custom_topk(score, self.ratio)
        perm = perm[(perm >= 0) & (perm < score.size(0))]

        mask = torch.zeros(score.size(0), dtype=torch.bool, device=x.device)
        mask[perm] = True

        x = x[mask]
        batch = batch[mask]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, mask)

        return x, edge_index, edge_attr, batch
