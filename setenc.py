import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

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
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
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

        #A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        A = torch.sigmoid(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V))
        #A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(dim_split), 2)
        #A = F.dropout(A, p=0.2, training=self.training)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, Q, K=None):
        if K is not None:
            return self.mab(Q=Q, K=K)
        return self.mab(Q=Q, K=Q)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class STEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers=2, num_inds=16, num_heads=4, num_outputs=1, ln=False, layer='sab', pool='pma'):
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
                self.encoder.append(stlayer(dim_in=dim_in, dim_out=dim_hidden//2 if num_layers > 1 else dim_hidden, num_heads=num_heads, ln=ln))
            else:
                self.encoder.append(stlayer(dim_in=dim_hidden//2 if num_layers > 1 else dim_hidden, dim_out=dim_hidden//2 if num_layers > 1 else
                                            dim_hidden, num_heads=num_heads, ln=ln))
        if pool == 'pma':
            self.encoder.append(PMA(dim=dim_hidden//2 if num_layers > 1 else dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln))
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
    # def __init__(self, sencoder='strans', layer='pma', dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_layers=1, num_heads=4, dim_proj=512, ln=True):
    def __init__(self, sencoder='dsets', layer='max', dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_layers=1, num_heads=4, dim_proj=512, ln=True):
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
        
        if sencoder == 'dsets':
            self.se_theta = DSEncoder(dim_in=dim_proj, dim_hidden=dim_proj, num_layers=num_layers, layer=layer)
        elif sencoder == 'strans':
            self.se_theta = STEncoder(num_layers=num_layers, dim_in=dim_proj, dim_hidden=dim_proj,  ln=ln, num_heads=num_heads)
        else:
            raise NotImplementedError
        self.dec = nn.Linear(dim_proj, dim_in)
        print(self.se_theta)

    def forward(self, x_real, x_context=None):
        r_theta_1 = self.r_theta_1(x_real)
        c_theta_1 = self.c_theta_1(x_context if x_context is not None else x_real.unsqueeze(1))
        x_rc = self.se_theta(torch.cat([r_theta_1.unsqueeze(1), c_theta_1], dim=1)).squeeze(1)
        x_rc = self.dec(x_rc)
        return x_rc

def get_mixer(args):
    return ContextMixer(dim_in=args.hidden_dim, dim_hidden=args.hidden_dim, num_outputs=args.num_outputs, ln=args.ln)
