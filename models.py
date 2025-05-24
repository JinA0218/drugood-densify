import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os

__all__ = ['get_model']

def get_model(args):
    if args.model == 'mlp':
        model = MLP(
                    in_features=args.in_features,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_outputs=args.num_outputs,
                    dropout=args.dropout,
                    mixing_layer=args.mixing_layer,
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
        
        # if os.environ.get("MIXING_X_DEFAULT", "xmix") != "xmix":
        #     self.fusion_mlp = torch.nn.Sequential(
        #                         torch.nn.Linear(hidden_dim * 2, hidden_dim),
        #                         torch.nn.ReLU()
        #                     )
        # else:
        #     self.fusion_mlp = None
    
    def forward(self, x, context=None, mixer_phi=None, embedding_list=None, label_list=None, embed_type=None, embed_test=None):
        if context is None:
            context = x.clone().unsqueeze(1)
        B, S, H = context.size()
        context = context.reshape(B*S, H) # NOTE originally view but changed to reshape
        
        B_X, S_X, H_X = None, None, None
        if embed_type != None and embed_type == "context_none":
            B_X, S_X, H_X = x.shape
            x = x.view(B_X * S_X, H_X)
        
        if os.environ.get('MIX_TYPE', 'SET') in ['SET', 'SET_NO_BILEVEL']:
            for i, theta in enumerate(self.mlp_theta):
                x = theta(x)
                if i <= self.mixing_layer:
                    context = theta(context)
                if mixer_phi != None and i == self.mixing_layer:
                    context = context.view(B, S, -1)
                    
                    if embed_type != None and embed_type == "context_none":
                        x = x.view(B_X, S_X, -1)
                    
                    if os.environ.get("MIXING_X_DEFAULT", "xmix") == "xmix":
                        x, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                    elif os.environ.get("MIXING_X_DEFAULT", "xmix") == "xmix_x" or os.environ.get("MIXING_X_DEFAULT", "xmix") == "mix_x":
                        raise Exception()
                        x_mixed, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                        print('#### x_mixed ', x_mixed.shape)
                        print('#### x bef concat', x.shape)
                        fusion_input = torch.cat([x, x_mixed], dim=1)
                        # x = self.fusion_mlp(fusion_input)
                        print('#### x aft concat', x.shape)
                        # breakpoint()
                    else:
                        raise Exception()
                
                if i == len(self.mlp_theta) - 2 and embed_type != None and "2nd_last" in embed_test:
                    embedding_list.append(x.detach().cpu())
                    if embed_type == "train_context":
                        label_list.append(torch.full((x.shape[0],), 0))
                    elif embed_type == "context_none":
                        label_list.append(torch.full((x.shape[0],), -1))
                    elif embed_type == "train_none":
                        label_list.append(torch.full((x.shape[0],), 1))
                    elif embed_type == "mvalid_none":
                        label_list.append(torch.full((x.shape[0],), 2))

                    elif embed_type == "ood1_none":
                        label_list.append(torch.full((x.shape[0],), 3))

                    elif embed_type == "ood2_none":
                        label_list.append(torch.full((x.shape[0],), 4))
                    else:
                        raise Exception()
            #context = context.view(B, S, -1)
            #x_mixed = mixer_phi(x_real=x, x_context=context)
            #y_hat= self.head_theta(x_mixed)
            
        elif os.environ.get('MIX_TYPE', 'SET') in ['MIXUP', 'MIXUP_BILEVEL', 'MANIFOLD_MIXUP', 'MANIFOLD_MIXUP_BILEVEL']:
            if os.environ.get('MIX_TYPE', 'SET') in ['MIXUP', 'MIXUP_BILEVEL']:
                context = context.view(B, S, -1)
                if embed_type != None and embed_type == "context_none":
                    x = x.view(B_X, S_X, -1)
                x, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                
                for i, theta in enumerate(self.mlp_theta):
                    x = theta(x)
                    # if i <= self.mixing_layer:
                    #     context = theta(context)
                        
                    # if mixer_phi != None and i == self.mixing_layer:
                    #     context = context.view(B, S, -1)
                        
                    #     if embed_type != None and embed_type == "context_none":
                    #         x = x.view(B_X, S_X, -1)
                        
                    #     if os.environ.get("MIXING_X_DEFAULT", "xmix") == "xmix":
                    #         x, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                    #     elif os.environ.get("MIXING_X_DEFAULT", "xmix") == "xmix_x" or os.environ.get("MIXING_X_DEFAULT", "xmix") == "mix_x":
                    #         raise Exception()
                    #         x_mixed, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                    #         print('#### x_mixed ', x_mixed.shape)
                    #         print('#### x bef concat', x.shape)
                    #         fusion_input = torch.cat([x, x_mixed], dim=1)
                    #         x = self.fusion_mlp(fusion_input)
                    #         print('#### x aft concat', x.shape)
                    #         # breakpoint()
                    #     else:
                    #         raise Exception()
                    
                    if i == len(self.mlp_theta) - 2 and embed_type != None and "2nd_last" in embed_test:
                        embedding_list.append(x.detach().cpu())
                        if embed_type == "train_context":
                            label_list.append(torch.full((x.shape[0],), 0))
                        elif embed_type == "context_none":
                            label_list.append(torch.full((x.shape[0],), -1))
                        elif embed_type == "train_none":
                            label_list.append(torch.full((x.shape[0],), 1))
                        elif embed_type == "mvalid_none":
                            label_list.append(torch.full((x.shape[0],), 2))

                        elif embed_type == "ood1_none":
                            label_list.append(torch.full((x.shape[0],), 3))

                        elif embed_type == "ood2_none":
                            label_list.append(torch.full((x.shape[0],), 4))
                        else:
                            raise Exception()
                
            elif os.environ.get('MIX_TYPE', 'SET') in ['MANIFOLD_MIXUP', 'MANIFOLD_MIXUP_BILEVEL']:
                for i, theta in enumerate(self.mlp_theta):
                    x = theta(x)
                    if i <= self.mixing_layer:
                        context = theta(context)
                    if i == self.mixing_layer:
                        context = context.view(B, S, -1)
                        
                        if embed_type != None and embed_type == "context_none":
                            x = x.view(B_X, S_X, -1)
                        
                        x, embedding_list, label_list = mixer_phi(x_real=x, x_context=context, embedding_list=embedding_list, label_list=label_list, embed_type=embed_type, embed_test=embed_test)
                        
                    if i == len(self.mlp_theta) - 2 and embed_type != None and "2nd_last" in embed_test:
                        embedding_list.append(x.detach().cpu())
                        if embed_type == "train_context":
                            label_list.append(torch.full((x.shape[0],), 0))
                        elif embed_type == "context_none":
                            label_list.append(torch.full((x.shape[0],), -1))
                        elif embed_type == "train_none":
                            label_list.append(torch.full((x.shape[0],), 1))
                        elif embed_type == "mvalid_none":
                            label_list.append(torch.full((x.shape[0],), 2))

                        elif embed_type == "ood1_none":
                            label_list.append(torch.full((x.shape[0],), 3))

                        elif embed_type == "ood2_none":
                            label_list.append(torch.full((x.shape[0],), 4))
                        else:
                            raise Exception()
            else:
                raise Exception()

        if embed_type != None and "lastlayer" in embed_test:
            embedding_list.append(x.detach().cpu())
            if embed_type == "train_context":
                label_list.append(torch.full((x.shape[0],), 0))
            elif embed_type == "context_none":
                label_list.append(torch.full((x.shape[0],), -1))
            elif embed_type == "train_none":
                label_list.append(torch.full((x.shape[0],), 1))
            elif embed_type == "mvalid_none":
                label_list.append(torch.full((x.shape[0],), 2))

            elif embed_type == "ood1_none":
                label_list.append(torch.full((x.shape[0],), 3))

            elif embed_type == "ood2_none":
                label_list.append(torch.full((x.shape[0],), 4))
            else:
                raise Exception()
            
        y_hat= self.head_theta(x)
        return y_hat, embedding_list, label_list
        
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
