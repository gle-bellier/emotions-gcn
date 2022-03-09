import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from src.model.conv_pool import ConvPool
from src.model.transformer_block import TransformerBlock


class Net(torch.nn.Module):
    def __init__(self, in_size, h_sizes, num_classes):
        super(Net, self).__init__()
        torch.manual_seed(12345)

        self.conv_top = TransformerBlock(in_size, h_sizes[0])
        self.pool_top = TopKPooling(h_sizes[0], ratio=2)

        self.gcn1 = GCNConv(h_sizes[0], h_sizes[1])
        self.gcn2 = GCNConv(h_sizes[1], h_sizes[2])

        self.conv_pools = nn.ModuleList([
            ConvPool(input_size, output_size, 3)
            for input_size, output_size in zip(h_sizes[2:-1], h_sizes[3:])
        ])
        self.conv_end = TransformerBlock(h_sizes[-1], h_sizes[-1])

        self.lin1 = nn.Linear(h_sizes[-1], num_classes * 2)
        self.lin2 = nn.Linear(num_classes * 2, num_classes)

        self.lr = nn.LeakyReLU()
        self.dp = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr, batch, batch_idx):
        # 1. Obtain node embeddings

        x = self.conv_top(x, edge_index, edge_attr)
        x = self.lr(x)

        x = self.dp(x)
        x = self.gcn1(x, edge_index)
        x = self.lr(x)

        x = self.dp(x)
        x = self.gcn2(x, edge_index)
        x = self.lr(x)

        for layer in self.conv_pools:
            x, edge_index, edge_attr, batch_idx = layer(
                x, edge_index, edge_attr, batch_idx)
            x = self.lr(x)

        x = self.conv_end(x, edge_index, edge_attr)
        x = self.lr(x)

        # 2. Readout layer
        x = gmp(x, batch_idx)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lr(x)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x
