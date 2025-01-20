import math
import torch
import torch.nn as nn
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
        A = F.dropout(A, p=0.2, training=self.training)
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
    def __init__(self, block, dim_in, dim_hidden, num_layers=2, num_inds=16, num_heads=4, num_outputs=1, ln=False):
        super(STEncoder, self).__init__()
        self.encoder = []
        self.encoder = nn.Sequential(
                SAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                #SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                #SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                #ISAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln),
                #ISAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln),
                #PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln),
                )
        #if block == 'sab':
        #    self.encoder.append(SAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, ln=ln))
        #    for i in range(num_layers):
        #        self.encoder.append(SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln))
        #elif block == 'isab':
        #    self.encoder.append(ISAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln))
        #    for i in range(num_layers):
        #        self.encoder.append(ISAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln))
        #else:
        #    raise NotImplementedError
        #self.encoder = nn.Sequential(*self.encoder)
        
    def forward(self, X):
        X = self.encoder(X)
        X = X.max(1).values
        return X

class DSEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers=4, layer='max'):
        super(DSEncoder, self).__init__()
        self.encoder = nn.Sequential(
                PermEquiMax(in_dim=dim_in, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PermEquiMax(in_dim=dim_hidden, out_dim=dim_hidden),
                #PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln),
                )
    
    def forward(self, X):
        X = self.encoder(X)
        #X, _ = X.mean(1)
        #X = X.sum(1)
        X, _ = X.max(1)
        return X

def get_st_encoder(block='sab', dim_in=2048, num_layers=2, dim_hidden=128, num_inds=16, num_outputs=1, num_heads=4, ln=True):
    encoder = []
    if block == 'sab':
        encoder.append(SAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, ln=ln))
        for i in range(num_layers):
            encoder.append(SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln))
    elif block == 'isab':
        encoder.append(ISAB(dim_in=dim_in, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln))
        for i in range(num_layers):
            encoder.append(ISAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln))
    else:
        raise NotImplementedError
    encoder.append(PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=num_outputs, ln=ln))
    return nn.Sequential(*encoder)

#class ContextMixer(nn.Module):
#    def __init__(self, dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_layers=2, num_heads=4, ln=True):
#        super(ContextMixer, self).__init__()
#        self.ln = ln
#        self.dim_in = dim_in
#        self.dim_in = dim_in
#        self.num_inds = num_inds
#        self.num_heads = num_heads
#        self.dim_hidden = dim_hidden
#        self.num_outputs = num_outputs
#        
#        #self.enc = STEncoder(block='sab', num_layers=2, dim_in=dim_in, dim_hidden=dim_hidden,  ln=ln, num_heads=num_heads)
#        #self.enc = get_st_encoder(
#        #        block='sab', 
#        #        dim_in=dim_in, 
#        #        num_outputs=num_layers, 
#        #        dim_hidden=dim_hidden,
#        #        num_inds=num_inds,
#        #        num_heads=num_heads,
#        #        ln=ln
#        #        )
#        self.enc = DSEncoder(dim_in=dim_in, dim_hidden=dim_hidden)
#
#    def forward(self, x_real, x_context):
#        X = torch.cat([x_real.unsqueeze(1), x_context], dim=1) if x_context is not None else torch.cat([x_real.unsqueeze(1), x_real.unsqueeze(1)],
#                                                                                                        dim=1)
#        X = self.enc(X)
#        return X.squeeze(1)

class ContextMixer(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=128, num_inds=16, num_outputs=1, num_layers=2, num_heads=4, ln=True):
        super(ContextMixer, self).__init__()
        dim = 512
        self.r_theta_1 = nn.Sequential(
                nn.Linear(in_features=dim_in, out_features=dim),
                nn.ReLU(),
                nn.Linear(in_features=dim, out_features=dim_hidden),
                nn.ReLU(),
                )
        
        self.c_theta_1 = nn.Sequential(
                nn.Linear(in_features=dim_in, out_features=dim),
                nn.ReLU(),
                nn.Linear(in_features=dim, out_features=dim_hidden),
                nn.ReLU(),
                )

        #self.se_theta = STEncoder(block='sab', num_layers=2, dim_in=dim_hidden, dim_hidden=dim_hidden,  ln=ln, num_heads=num_heads)
        self.se_theta = DSEncoder(dim_in=dim_hidden, dim_hidden=dim_hidden)
        self.dec = nn.Linear(dim_hidden, dim_in)

    def forward(self, x_real, x_context=None):
        r_theta_1 = self.r_theta_1(x_real)
        c_theta_1 = self.c_theta_1(x_context if x_context is not None else x_real.unsqueeze(1))
        x_rc = self.se_theta(torch.cat([r_theta_1.unsqueeze(1), c_theta_1], dim=1)).squeeze(1)
        x_rc = self.dec(x_rc)
        return x_rc

def get_mixer(args):
    return ContextMixer(dim_in=args.hidden_dim, dim_hidden=args.hidden_dim, num_outputs=args.num_outputs, ln=args.ln)
