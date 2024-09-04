import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MLP', 'MLP2']

class MLP(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=32, num_layers=2, num_outputs=2, dropout=0.2, batchnorm=True):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        
        self.mlp_theta = []
        for i in range(num_layers):
            dim = in_features if i==0 else hidden_dim
            self.mlp_theta.append(nn.Linear(in_features=dim, out_features=hidden_dim))
            if self.batchnorm:
                self.mlp_theta.append(nn.BatchNorm1d(num_features=hidden_dim))
            self.mlp_theta.append(nn.ReLU(inplace=True))
        if  self.dropout > 0.0:
            self.mlp_theta.append(nn.Dropout(p=self.dropout))
        self.mlp_theta = nn.Sequential(*self.mlp_theta)
        self.head_theta = nn.Linear(in_features=hidden_dim, out_features=num_outputs)

    def forward(self, x, context=None, mixer_phi=None):
        z_x = self.mlp_theta(x)
        if context is not None:
            z_c = self.mlp_theta(context)
            z_xc = mixer_phi(X=torch.cat([z_x.unsqueeze(1), z_c[:x.size(0), :].unsqueeze(1)], dim=1))
        else:
            if mixer_phi is not None: #NOTE: This is for when we train the plain mlp without contextmixer
                z_xc = mixer_phi(X=z_x.unsqueeze(1))
        y_hat = self.head_theta(z_xc)
        return y_hat

class MLP2(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=32, num_layers=2, num_outputs=2, dropout=0.2, batchnorm=True):
        super(MLP2, self).__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        
        self.mlp_theta = []
        for i in range(num_layers):
            dim = in_features if i==0 else hidden_dim
            self.mlp_theta.append(nn.Linear(in_features=dim, out_features=hidden_dim))
            if self.batchnorm:
                self.mlp_theta.append(nn.BatchNorm1d(num_features=hidden_dim))
            self.mlp_theta.append(nn.ReLU(inplace=True))
        if  self.dropout > 0.0:
            self.mlp_theta.append(nn.Dropout(p=self.dropout))
        self.mlp_theta = nn.Sequential(*self.mlp_theta)
        
        self.mlp_theta_c = []
        for i in range(num_layers):
            dim = in_features if i==0 else hidden_dim
            self.mlp_theta_c.append(nn.Linear(in_features=dim, out_features=hidden_dim))
            if self.batchnorm:
                self.mlp_theta_c.append(nn.BatchNorm1d(num_features=hidden_dim))
            self.mlp_theta_c.append(nn.ReLU(inplace=True))
        if  self.dropout > 0.0:
            self.mlp_theta_c.append(nn.Dropout(p=self.dropout))
        self.mlp_theta_c = nn.Sequential(*self.mlp_theta_c)
        
        self.mlp_theta_xc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim),
                nn.ReLU(inplace=True),
                )

        self.head_theta = nn.Linear(in_features=hidden_dim, out_features=num_outputs)

    def forward(self, x, context=None, mixer_phi=None):
        z_x = self.mlp_theta(x)
        
        if context is None:
            context = x.clone()

        #y_hat_xc = None
        #if context is not None:
        z_c = self.mlp_theta_c(context[:x.size(0), :])
        z_xc = mixer_phi(X=torch.cat([z_x.unsqueeze(dim=1), z_c.unsqueeze(dim=1)], dim=1))
        z_hat_xc = self.mlp_theta_xc(z_xc)
        y_hat_xc = self.head_theta(z_hat_xc)
        return y_hat_xc
        #y_hat_x = self.head_theta(z_x)
        #return y_hat_x
