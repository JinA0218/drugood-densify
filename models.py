import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['get_model']

def get_model(args):
    if args.model == 'mlp':
        model = MLP(
                    in_features=args.in_features,
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
                 num_layers=1, 
                 num_outputs=2, 
                 dropout=0.2,
                 mixing_layer=0,
                 batchnorm=True
                 ):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.mixing_layer = mixing_layer
        
        assert mixing_layer < num_layers, 'Mixing layer cannot be larger than number of layers'

        self.mlp_theta = []
        for i in range(num_layers):
            dim = in_features if i==0 else hidden_dim
            mlp = []
            mlp.append(nn.Linear(in_features=dim, out_features=hidden_dim))
            if self.batchnorm:
                mlp.append(nn.BatchNorm1d(num_features=hidden_dim))
            mlp.append(nn.ReLU(inplace=True))
            
            if (i + 1) == self.num_layers:
                if  self.dropout > 0.0:
                    mlp.append(nn.Dropout(p=self.dropout))
            self.mlp_theta.append(nn.Sequential(*mlp))
        self.mlp_theta = nn.ModuleList(self.mlp_theta)
        self.head_theta = nn.Linear(in_features=hidden_dim, out_features=num_outputs)

    def forward(self, x, context=None, mixer_phi=None):
        if context is None:
            context = x.clone().unsqueeze(1)
        B, S, H = context.size()
        context = context.view(B*S, H)
        for i, theta in enumerate(self.mlp_theta):
            x = theta(x)
            if i <= self.mixing_layer:
                context = theta(context)
            if i == self.mixing_layer:
                context = context.view(B, S, -1)
                x = mixer_phi(x_real=x, x_context=context)
        #context = context.view(B, S, -1)
        #x_mixed = mixer_phi(x_real=x, x_context=context)
        #y_hat= self.head_theta(x_mixed)
        y_hat= self.head_theta(x)
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
