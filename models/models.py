import torch.nn as nn
from models import MLP, MLP2, ContextMixer

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
    elif args.model == 'mlp2':
        model = MLP2(in_features=args.in_features, \
                hidden_dim=args.hidden_dim, \
                num_layers=args.num_layers, \
                num_outputs=args.num_outputs, \
                dropout=args.dropout, \
                batchnorm=args.batchnorm)
    else:
        raise NotImplementedError
    
    if args.initialize_weights:
            initialize_weights(model=model)

    mixer_phi = None
    if args.mixer_phi:
        hidden_dim = args.hidden_dim if args.model == 'mlp' else 2*args.hidden_dim
        mixer_phi = ContextMixer(dim_in=args.hidden_dim, dim_hidden=hidden_dim, num_inds=32, num_outputs=1, num_heads=4, ln=True)

        if args.initialize_weights:
            initialize_weights(model=mixer_phi)

    return model, mixer_phi
