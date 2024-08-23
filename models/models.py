import torch.nn as nn
from models import MLP, ContextMixer

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def get_model(args):
    if args.model == 'mlp':
        model = MLP(in_features=args.in_features, \
                hidden_dim=args.hidden_dim, \
                num_layers=args.num_layers, \
                num_outputs=args.num_outputs, \
                dropout=args.dropout, \
                batchnorm=args.batchnorm)
        #initialize_weights(model=model)
    else:
        raise NotImplementedError
    
    contextmixer = None
    if args.contextmixer:
        #TODO: Move remaining parameters to args
        contextmixer = ContextMixer(dim_in=args.hidden_dim, dim_hidden=args.hidden_dim, num_inds=32, num_outputs=1, num_heads=4, ln=True)
    return model, contextmixer
