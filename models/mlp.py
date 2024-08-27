import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=32, num_layers=2, num_outputs=2, dropout=0.2, batchnorm=True):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        
        self.mlp = []
        for i in range(num_layers):
            dim = in_features if i==0 else hidden_dim
            self.mlp.append(nn.Linear(in_features=dim, out_features=hidden_dim))
            if self.batchnorm:
                self.mlp.append(nn.BatchNorm1d(num_features=hidden_dim))
            self.mlp.append(nn.ReLU(inplace=True))
        if  self.dropout > 0.0:
            self.mlp.append(nn.Dropout(p=self.dropout))
        self.mlp = nn.Sequential(*self.mlp)
        self.head = nn.Linear(in_features=hidden_dim, out_features=num_outputs)

    def forward(self, x, context=None, contextmixer=None):
        x = self.mlp(x)
        if context is not None:
            context = self.mlp(context)
            x = contextmixer(X=torch.cat([x.unsqueeze(1), context[:x.size(0), :].unsqueeze(1)], dim=1))
        else:
            if contextmixer is not None: #NOTE: This is for when we train the plain mlp without contextmixer
                x = contextmixer(X=x.unsqueeze(1))
        x = self.head(x)
        return x

if __name__ == '__main__':
    mlp = MLP()
    print(mlp)
    x = torch.rand(2, 2048)
    x = mlp(x)
    print(x.size())
