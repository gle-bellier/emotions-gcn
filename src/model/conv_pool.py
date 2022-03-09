import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling

from src.model.transformer_block import TransformerBlock


class ConvPool(nn.Module):
    def __init__(self, in_size, out_size, ratio):
        super(ConvPool, self).__init__()
        self.conv2 = TransformerBlock(in_size, out_size)
        self.pool = TopKPooling(out_size, ratio=ratio)
        self.bn = BatchNorm1d(out_size)

    def forward(self, x, edge_index, edge_attr, batch_idx):
        # 1. Obtain node embeddings
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x, edge_index, edge_attr, batch_idx, _, _ = self.pool(
            x, edge_index, edge_attr, batch_idx)

        x = self.bn(x)
        return x, edge_index, edge_attr, batch_idx
