import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv
from dgl.nn.pytorch import HeteroGraphConv
from contextlib import contextmanager
from tpp_pytorch_extension.gnn.gat import fused_GAT as tpp_gat
from tpp_pytorch_extension.gnn.common import gnn_utils

use_opt_mlp = False
linear = None

@contextmanager
def opt_impl(enable=True, use_bf16=False):
    try:
        global GATConv
        global linear
        global use_opt_mlp
        orig_GATConv = GATConv
        linear = nn.Linear
        try:
            if enable:
                use_opt_mlp = enable
                if use_bf16:
                    GATConv = tpp_gat.GATConvOptBF16
                    linear = tpp_gat.LinearOutBF16
                else:
                    GATConv = tpp_gat.GATConvOpt
                    linear = tpp_gat.LinearOut
            yield
        finally:
            GATConv = orig_GATConv
            linear = nn.Linear
    except ImportError as e:
        pass

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        for _ in range(num_layers-2):
            self.layers.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
        self.layers.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, h_feats))
        for _ in range(num_layers-2):
            self.layers.append(GraphConv(h_feats, h_feats))
        self.layers.append(GraphConv(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads, num_layers=2, dropout=0.2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, h_feats, num_heads))
        for _ in range(num_layers-2):
            self.layers.append(GATConv(h_feats * num_heads, h_feats, num_heads))
        self.layers.append(GATConv(h_feats * num_heads, num_classes, num_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            if l < len(self.layers) - 1:
                h = layer(block, (h, h_dst)).flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = layer(block, (h, h_dst)).mean(1)  
        return h


class RGCN(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(in_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GraphConv(h_feats, h_feats)
                for etype in etypes}, aggregate='mean'))
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(h_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)  

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])

class RSAGE(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(in_feats, h_feats, 'gcn')
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: SAGEConv(h_feats, h_feats, 'gcn')
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(h_feats, h_feats, 'gcn')
            for etype in etypes}))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)  

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])

class RGAT(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers, n_heads, activation, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        if not use_opt_mlp: activation = None
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads, activation=activation, feat_drop=dropout)
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads, activation=activation, feat_drop=dropout)
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, n_heads, activation=None)
            for etype in etypes}))
        if not use_opt_mlp:
            self.dropout = nn.Dropout(dropout)
        self.linear = linear(h_feats, num_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            #h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1 and not use_opt_mlp:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])   
