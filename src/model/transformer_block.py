import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d

from torch_geometric.nn import TransformerConv


class TransformerBlock(nn.Module):
    def __init__(self, in_size, out_size, n_heads=3):
        super(TransformerBlock, self).__init__()
        self.conv1 = TransformerConv(in_size,
                                     out_size,
                                     heads=n_heads,
                                     edge_dim=1,
                                     beta=True)

        self.transf1 = Linear(out_size * n_heads, out_size)
        self.bn1 = BatchNorm1d(out_size)
        self.dp = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        # Initial transformation

        x = self.dp(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        return x
