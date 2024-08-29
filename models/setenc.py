import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ContextMixer']

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
        A = torch.sigmoid(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V))
        #A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(dim_split), 2)
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
    def __init__(self, dim_in, dim_hidden, num_inds=16, num_heads=4, num_outputs=1, ln=False):
        super(STEncoder, self).__init__()
        ln = False
        self.encoder = nn.Sequential(
                #SAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                #SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                ISAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln),
                ISAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln),
                #PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln),
                )

    def forward(self, X):
        X = self.encoder(X)
        return X.sum(dim=1)
        #return self.encoder(X).squeeze(1)

class DSEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_inds=16, num_heads=4, num_outputs=1, ln=True):
        super(DSEncoder, self).__init__()
        self.encoder = nn.Sequential(
                PermEquiSum(in_dim=dim_in, out_dim=dim_hidden),
                #PermEquiMean(in_dim=dim_in, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_in, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln),
                )

    def forward(self, X):
        X = self.encoder(X)
        #X, _ = X.mean(1)
        X = X.sum(1)
        return X

class ContextMixer(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_heads=4, ln=True):
        super(ContextMixer, self).__init__()
        self.ln = ln
        self.dim_in = dim_in
        self.dim_in = dim_in
        self.num_inds = num_inds
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.num_outputs = num_outputs
        
        #self.enc = STEncoder(dim_in=dim_in, dim_hidden=dim_hidden)
        self.enc = DSEncoder(dim_in=dim_in, dim_hidden=dim_hidden)

        if dim_in != dim_hidden:
            self.linear = nn.Sequential(
                    nn.Linear(in_features=dim_hidden, out_features=dim_in),
                    )
        else:
            self.linear = None

    def forward(self, X):
        X = self.enc(X)
        if self.linear is not None:
            X = self.linear(X)
        return X.squeeze(1)
