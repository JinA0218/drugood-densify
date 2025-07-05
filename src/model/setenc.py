import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import os

__all__ = ['get_mixer']

class PermEquiMax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x

class PermEquiMean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x

class PermEquiSum(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x):
        xm = x.sum(1, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, sigmoid=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.sigmoid = sigmoid
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if not self.sigmoid:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
            A = F.dropout(A, p=0.5, training=self.training)
        else:
            A = torch.sigmoid(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V))

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, sigmoid=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, sigmoid=sigmoid)

    def forward(self, Q, K=None):
        if K is not None:
            return self.mab(Q=Q, K=K)
        return self.mab(Q=Q, K=Q)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, sigmoid=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, sigmoid=sigmoid)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, sigmoid=sigmoid)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, sigmoid=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, sigmoid=sigmoid)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class STEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers=2, num_inds=16, num_heads=4, num_outputs=1, ln=False, layer='sab', pool='pma', mab_sigmoid=True, pma_sigmoid=True):
        super(STEncoder, self).__init__()
        self.ln = ln
        self.pool = pool
        self.layer = layer
        self.dim_in = dim_in
        self.num_inds = num_inds
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        
        if layer == 'sab':
            stlayer = SAB
        elif layer == 'isab':
            stlayer = partial(ISAB, num_inds=num_inds)
        else:
            raise NotImplementedError

        self.encoder = []
        for i in range(num_layers):
            if i == 0:
                self.encoder.append(stlayer(dim_in=dim_in, dim_out=dim_hidden//2 if num_layers > 1 else dim_hidden, num_heads=num_heads, ln=ln, sigmoid=mab_sigmoid))
            else:
                self.encoder.append(stlayer(dim_in=dim_hidden//2 if num_layers > 1 else dim_hidden, dim_out=dim_hidden//2 if num_layers > 1 else
                                            dim_hidden, num_heads=num_heads, ln=ln, sigmoid=mab_sigmoid))
        if pool == 'pma':
            self.encoder.append(PMA(dim=dim_hidden//2 if num_layers > 1 else dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln, sigmoid=pma_sigmoid))
        self.encoder = nn.Sequential(*self.encoder)
        self.proj = nn.Linear(in_features=dim_hidden//2 if num_layers > 1 else dim_hidden, out_features=dim_hidden)

    def aggregate(self, X):
        if self.pool == 'max':
            X, _ = X.max(dim=1)
        elif self.pool == 'mean':
            X = X.mean(dim=1)
        elif self.pool == 'sum':
            X = X.sum(dim=1)
        return X
   
    def forward(self, X):
        X = self.encoder(X)
        if self.pool in ['mean', 'max', 'sum']:
            X = self.aggregate(X=X)
        X = self.proj(X)
        return X

        # X = self.encoder(X)
        # return X.max(1).values

class DSEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers=1, layer='max'):
        super(DSEncoder, self).__init__()
        self.layer = layer
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        
        if layer == 'max':
            dslayer = PermEquiMax
        elif layer == 'sum':
            dslayer = PermEquiSum
        elif layer == 'mean':
            dslayer = PermEquiMean
        else:
            raise NotImplementedError

        self.encoder = []
        for i in range(num_layers):
            if i == 0:
                self.encoder.append(dslayer(in_dim=dim_in, out_dim=dim_hidden))
            else:
                self.encoder.append(dslayer(in_dim=dim_hidden, out_dim=dim_hidden))
        self.encoder = nn.Sequential(*self.encoder)
    
    def aggregate(self, X):
        if self.layer == 'max':
            X, _ = X.max(dim=1)
        elif self.layer == 'mean':
            X = X.mean(dim=1)
        elif self.layer == 'sum':
            X = X.sum(dim=1)
        return X

    def forward(self, X):
        X = self.encoder(X)
        X = self.aggregate(X=X)
        return X

class ContextMixer(nn.Module):
    def __init__(self, sencoder='strans', layer='pma', dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_layers=1, num_heads=4, dim_proj=512, ln=True):
        super(ContextMixer, self).__init__()
        self.r_theta_1 = nn.Sequential(
                nn.Linear(in_features=dim_in, out_features=dim_proj),
                nn.ReLU(),
                nn.Linear(in_features=dim_proj, out_features=dim_proj),
                nn.ReLU(),
                )
        
        self.c_theta_1 = nn.Sequential(
                nn.Linear(in_features=dim_in, out_features=dim_proj),
                nn.ReLU(),
                nn.Linear(in_features=dim_proj, out_features=dim_proj),
                nn.ReLU(),
                )
        
        self.sencoder_layer = layer
        
        if sencoder == 'dsets':
            print("loading deepsets")
            self.se_theta = DSEncoder(dim_in=dim_proj, dim_hidden=dim_proj, num_layers=num_layers, layer=layer)
        elif sencoder == 'strans':
            print(f"loading set transformer {layer=}")

            # "pma" is used for merck which used sigmoid for all attention layers.
            # zinc experiments used softmax + max pooling
            if layer == "pma":
                mab_sigmoid = True
                pma_sigmoid = True
            else:
                mab_sigmoid = False
                pma_sigmoid = False

            self.se_theta = STEncoder(
                num_layers=num_layers, dim_in=dim_proj, dim_hidden=dim_proj,
                ln=ln, num_heads=num_heads, pool=layer,
                mab_sigmoid=mab_sigmoid, pma_sigmoid=pma_sigmoid
            )
        else:
            raise NotImplementedError
        self.dec = nn.Linear(dim_proj, dim_in)
        print(self.se_theta)

    def mixup_with_context(self, x=None, context=None, sencoder_layer=None, embed_type=None, ):
        """
        Mix each (B, H) input with its own (B, S, H) context.
        Returns: mixed_x of shape (B, H)
        """
        B, S, H = context.shape
        
        if not (embed_type != None and embed_type == "context_none"):
            x = x.unsqueeze(1)  # (B, 1, H)

        # Uniformly sample lambda in [0, 1] per sample
        lam = torch.rand((B, 1, 1), device=x.device)  # (B, 1, 1)

        mixed = lam * x + (1 - lam) * context  # (B, S, H)

        # Reduce along context dimension
        if sencoder_layer == 'mean':
            return mixed.mean(dim=1)  # (B, H)
        elif sencoder_layer == 'max':
            return mixed.max(dim=1).values  # (B, H)
        elif sencoder_layer == 'sum':
            return mixed.sum(dim=1)  # (B, H)
        else:
            raise ValueError(f"Unsupported sencoder_layer: {sencoder_layer}")

    
    def forward(self, x_real, x_context=None, embedding_list=None, label_list=None, embed_type=None, embed_test=None):
        if embed_type != None and embed_type == "context_none": # TODO BE CAREFUL
            if os.environ.get('MIX_TYPE', 'SET') in ['SET', 'SET_NO_BILEVEL'] or os.environ.get('MIX_TYPE', 'SET') in ['MANIFOLD_MIXUP', 'MANIFOLD_MIXUP_BILEVEL']:
                # print("SHOULD NOT ENCOUNTER DURING TRAINING !!!!!!")
                c_theta_1_x = self.c_theta_1(x_real)
                c_theta_1_c = self.c_theta_1(x_context)
                
                if os.environ.get('MIX_TYPE', 'SET') in ['SET', 'SET_NO_BILEVEL']:
                    x_rc = self.se_theta(torch.cat([c_theta_1_x, c_theta_1_c], dim=1)).squeeze(1)
                else:
                    # print('c_theta_1_x ', c_theta_1_x.shape)
                    # print('c_theta_1_x ', c_theta_1_x.squeeze(1).shape)
                    x_rc = self.mixup_with_context(x=c_theta_1_x, context=c_theta_1_c, sencoder_layer=self.sencoder_layer, embed_type=embed_type)
            elif os.environ.get('MIX_TYPE', 'SET') in ['MIXUP', 'MIXUP_BILEVEL']:
                # print('x_real ', x_real.shape)
                # print('x_context ', x_context.shape)
                x_rc = self.mixup_with_context(x=x_real, context=x_context, sencoder_layer=self.sencoder_layer, embed_type=embed_type)
                # print('x_rc ', x_rc.shape)
                return x_rc, embedding_list, label_list
            else:
                raise Exception()
            
        else:
            if os.environ.get('MIX_TYPE', 'SET') in ['MIXUP', 'MIXUP_BILEVEL']:
                x_rc = self.mixup_with_context(x=x_real, context=x_context, sencoder_layer=self.sencoder_layer)
                
                # print('###### x_rc ', x_rc.shape)
                if embed_type != None and "setenc" in embed_test:
                    embedding_list.append(x_rc.detach().cpu())
                    if embed_type == "train_context":
                        label_list.append(torch.full((x_rc.shape[0],), 0))
                    elif embed_type == "context_none":
                        label_list.append(torch.full((x_rc.shape[0],), -1))
                    elif embed_type == "train_none":
                        label_list.append(torch.full((x_rc.shape[0],), 1))
                    elif embed_type == "mvalid_none":
                        label_list.append(torch.full((x_rc.shape[0],), 2))

                    elif embed_type == "ood1_none":
                        label_list.append(torch.full((x_rc.shape[0],), 3))

                    elif embed_type == "ood2_none":
                        label_list.append(torch.full((x_rc.shape[0],), 4))
                    else:
                        raise Exception()
                
                return x_rc, embedding_list, label_list
            else:
                r_theta_1 = self.r_theta_1(x_real)
                c_theta_1 = self.c_theta_1(x_context if x_context is not None else x_real.unsqueeze(1))

                if os.environ.get('MIX_TYPE', 'SET') in ['SET', 'SET_NO_BILEVEL']:
                    if os.environ.get("MIXING_X_DEFAULT", "xmix") == "mix_x": # also consider context_none
                        raise Exception()
                        x_rc = self.se_theta(c_theta_1).squeeze(1)
                        # print('NO EFFECT OF SET!!!!')
                        print('>> c_theta_1 ', c_theta_1.shape)
                        print('>> x_rc bef ', self.se_theta(c_theta_1).shape)
                        print('>> x_rc ', x_rc.shape)
                        # breakpoint()
                    else:
                        x_rc = self.se_theta(torch.cat([r_theta_1.unsqueeze(1), c_theta_1], dim=1)).squeeze(1)
                    
                elif os.environ.get('MIX_TYPE', 'SET') in ['MANIFOLD_MIXUP', 'MANIFOLD_MIXUP_BILEVEL']:
                    x_rc = self.mixup_with_context(x=r_theta_1, context=c_theta_1, sencoder_layer=self.sencoder_layer)
                else:
                    raise Exception()

        
        # print('###### x_rc ', x_rc.shape)
        if embed_type != None and "setenc" in embed_test:
            embedding_list.append(x_rc.detach().cpu())
            if embed_type == "train_context":
                label_list.append(torch.full((x_rc.shape[0],), 0))
            elif embed_type == "context_none":
                label_list.append(torch.full((x_rc.shape[0],), -1))
            elif embed_type == "train_none":
                label_list.append(torch.full((x_rc.shape[0],), 1))
            elif embed_type == "mvalid_none":
                label_list.append(torch.full((x_rc.shape[0],), 2))

            elif embed_type == "ood1_none":
                label_list.append(torch.full((x_rc.shape[0],), 3))

            elif embed_type == "ood2_none":
                label_list.append(torch.full((x_rc.shape[0],), 4))
            else:
                raise Exception()
            
        x_rc = self.dec(x_rc)
        return x_rc, embedding_list, label_list

def get_mixer(args):
    return ContextMixer(
        sencoder=args.sencoder,
        layer=args.sencoder_layer,
        dim_in=args.hidden_dim,
        dim_hidden=args.hidden_dim,
        num_outputs=args.num_outputs,
        ln=args.ln
    )
