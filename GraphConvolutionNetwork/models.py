import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

'''The given code defines the forward() method for the PyTorch module "GCN", which implements a two-layer graph convolutional neural network for node classification. Here's a brief overview of the code:

The method takes two input arguments: "x" (a PyTorch tensor), which is the feature matrix for the input graph; and "adj" (a PyTorch sparse tensor), which is the adjacency matrix for the input graph.
The method passes the feature matrix "x" through the first GraphConvolution layer "gc1", followed by a ReLU activation function.
The method applies dropout to the output of the first layer using the PyTorch function F.dropout, with the specified dropout rate.
The method passes the output of the first layer through the second GraphConvolution layer "gc2".
The method applies the log softmax function to the output of the second layer along the "dim=1" dimension, which normalizes the predicted class probabilities for each node.
The method returns the predicted class probabilities as a PyTorch tensor.'''


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)