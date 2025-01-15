import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['get_model']

def get_model(args):
    if args.model == 'mlp':
        model = MLP(in_features=args.in_features,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_outputs=args.num_outputs,
                    dropout=args.dropout,
                    batchnorm=args.batchnorm,
                    )
    else:
        raise NotImplementedError
    
    if args.initialize_weights:
        model = initialize_weights(model=model)

    return model

class MLP(nn.Module):
    def __init__(self, 
                 in_features=2048, 
                 hidden_dim=32, 
                 num_layers=2, 
                 num_outputs=2, 
                 dropout=0.2, 
                 batchnorm=True
                 ):
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

    def forward(self, 
                x, 
                context=None, 
                mixer_phi=None
                ):
        z_x = self.mlp_theta(x)
        if context is not None:
            z_c = self.mlp_theta(context)
            z_xc = mixer_phi(X=torch.cat([z_x.unsqueeze(1), z_c[:x.size(0), :].unsqueeze(1)], dim=1))
        else:
            if mixer_phi is not None: #NOTE: This is for when we train the plain mlp without contextmixer
                z_xc = mixer_phi(X=z_x.unsqueeze(1))
            else:
                z_xc = z_x
        y_hat = self.head_theta(z_xc)
        return y_hat

def initialize_weights(model):
    def initialize(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
    print('Initializing model parameters')
    model.apply(initialize)
    return model
