import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from setenc import ContextMixer
from arguments import get_arguments
from utils import set_seed, get_optimizer, InfIterator

split_types = {
        'spectral': 'spectral_split',
        'random': 'random_split',
        'scaffold': 'scaffold_split',
        'weight': 'mw_split',
        }

fingerprints = {
        'ecfp': 'ec_bit_fp',
        'rdkit': 'rdkit_bit_fp',
        }

class ZINC(Dataset):
    def __init__(self, fingerprint='ecfp', batchsize=64):
        self.batchsize = batchsize
        self.fingerprint = fingerprint
        self.data = self.load_data()

    def load_data(self):
        path = os.path.join('data/ZINC', self.fingerprint)
        filepaths = glob.glob(os.path.join(path, '*.pth'))
        return filepaths

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        randperm = torch.randperm(self.batchsize)
        return data[randperm, :].float()
    
    def __len__(self):
        return len(self.data)

class AntiMalaria(Dataset):
    ''' 
    Split Types:
            - spectral split
            - random split
            - scaffold split
            - weight split

    Fingerprint Types:
            - ECFP
            - rdkitFP    TODO: Check original jax code to extract this version if available.
    '''


    def __init__(self, root='data', split='train', split_type='spectral', fingerprint='ecfp'):
        self.root = root
        self.split = split
        self.split_type = split_type
        self.fingerprint = fingerprint

        self.features, self.labels = self.load_dataset()
        self.classweights = torch.from_numpy(
                np.asarray(compute_class_weight(class_weight='balanced', classes=np.unique(self.labels.numpy()), y=self.labels.numpy()))
                                             )

    def load_dataset(self):
        datapath = os.path.join(
                self.root, 
                'antimalarial_data_processed', 
                split_types[self.split_type], 
                fingerprints[self.fingerprint],
                '2', 
                '{}.pth'.format(self.split)
                )
        data = torch.load(datapath)
        features, labels = data['x'].float(), data['y']
        return features, labels

    def __getitem__(self, index):
        x, y_i = self.features[index], self.labels[index]
        if self.split == 'train':
            y = torch.zeros(2)
            y[int(y_i.item())] = 1.0
        else:
            y = y_i
        return x, y.float()

    def __len__(self):
        return self.features.size(0)

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def get_dataset(args):
    if args.dataset == 'antimalaria':
        trainset = AntiMalaria(
                root=args.root, 
                split='train', 
                split_type=args.split_type, 
                fingerprint=args.fingerprint
                )
        validset = AntiMalaria(
                root=args.root, 
                split='valid', 
                split_type=args.split_type, 
                fingerprint=args.fingerprint
                )
        testset = AntiMalaria(
                root=args.root, 
                split='test', 
                split_type=args.split_type, 
                fingerprint=args.fingerprint
                )
    else:
        raise NotImplementedError
    
    #g = torch.Generator()
    #g.manual_seed(0)
    trainloader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            #worker_init_fn=seed_worker, 
            #generator=g, 
            shuffle=True, 
            pin_memory=True
            )
    validloader = DataLoader(
            validset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            #worker_init_fn=seed_worker, 
            #generator=g,
            shuffle=False, 
            pin_memory=True
            )

    testloader = DataLoader(testset, \
            batch_size=args.batch_size, \
            num_workers=args.num_workers, \
            #worker_init_fn=seed_worker, \
            #generator=g,
            shuffle=False, pin_memory=True)
    
    contextloader = None
    if args.mixer_phi:
        contextloader = ZINC(batchsize=args.batch_size, fingerprint=args.fingerprint)
    return trainloader, validloader, testloader, contextloader

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

class MLP2(nn.Module):
    def __init__(self, 
                 in_features=2048, 
                 hidden_dim=32, 
                 num_layers=2, 
                 num_outputs=2, 
                 dropout=0.2, 
                 batchnorm=True
                 ):
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

    def forward(self, 
                x, 
                context=None, 
                mixer_phi=None
                ):
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
        mixer_phi = ContextMixer(
                dim_in=args.hidden_dim, 
                dim_hidden=hidden_dim, 
                num_inds=32, 
                num_outputs=1, 
                num_heads=4, 
                ln=True
                )

        if args.initialize_weights:
            initialize_weights(model=mixer_phi)

    return model, mixer_phi

class Trainer:
    def __init__(self, epochs=500, \
            model=None, \
            mixer_phi=None, \
            optimizer=None, \
            testloader=None, \
            trainloader=None, \
            validloader=None, \
            contextloader=None, \
            optimizermixer=None, \
            args=None):
        self.args=args
        self.model = model
        self.optimizer = optimizer
        self.testloader = testloader
        self.trainloader = trainloader
        self.validloader = validloader
        self.mixer_phi = mixer_phi
        self.contextloader = contextloader
        self.optimizermixer = optimizermixer
        
    def train(self):
        losses = []
        self.model.train()
        self.optimizer.train()
        for x,y in self.trainloader:
            y_hat = self.model(x=x.to(self.args.device))
            loss = F.cross_entropy(y_hat, y.to(self.args.device))
            losses.append(loss.item())           
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)

    def test(self, dataloader, mixer_phi=None):
        self.model.eval()
        self.optimizer.eval()
        if mixer_phi is not None:
            mixer_phi.eval()
        
        with torch.no_grad():
            preds, labels = [], []
            for x, y in dataloader:
                y_hat = self.model(x=x.to(self.args.device), mixer_phi=mixer_phi)
                pred = torch.softmax(y_hat, dim=-1)
                preds.append(pred[:, 1])
                labels.append(y)
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
        #self.model.eval()
        nll = log_loss(y_true=labels.cpu(), y_pred=preds.cpu())
        preds = preds.cpu().numpy().tolist()
        labels = labels.bool().cpu().numpy().tolist()
        
        #TODO: check the definitions of the auc_roc and brier scores
        auc_roc = roc_auc_score(labels, preds)
        brier = brier_score_loss(labels, preds)
        return nll, auc_roc, brier
    
    def train_mixer_phi(self):
        def approxInverseHVP(v, f, w, i=5, alpha=0.1):
            p = [v_.clone().detach() for v_ in v]
            for j in range(i):
                grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)
                v = [v_ - alpha * g_ for v_,g_ in zip(v, grad)]
                p = [v_ + p_ for v_,p_ in zip(v, p)]
            return [alpha * p_ for p_ in p]

        def hypergradients(L_V, L_T, lmbda, w, i=5, alpha=0.1):
            v1 = torch.autograd.grad(L_V, w(), retain_graph=True)

            d_LT_dw = torch.autograd.grad(L_T, w(), create_graph=True)
            
            v2 = approxInverseHVP(v=v1, f=d_LT_dw, w=w, i=i, alpha=alpha)
            
            v3 = torch.autograd.grad(d_LT_dw, lmbda(), grad_outputs=v2, retain_graph=True)
            d_LV_dlmbda = torch.autograd.grad(L_V, lmbda())
            return [d - v for d,v in zip(d_LV_dlmbda, v3)]
            
        def train_mixer(model, optimizer, mixer_phi, x, y, context, device, interp_loss=False):
            model.train(); mixer_phi.train()
            if optimizer is not None:
                optimizer.train()
            
            train_loss, train_acc = 0.0, 0.0

            x, y, context = x.to(device), y.to(device), context.to(device)
            
            #1. Mix context with labeled sample x
            y_hat_mixed = model(x=x, context=context, mixer_phi=mixer_phi)
            loss = F.cross_entropy(y_hat_mixed, y, weight=self.trainloader.dataset.classweights.to(self.args.device))
            
            #2. Pass unmixed sample through model
            #y_hat = model(x=x, context=None, mixer_phi=mixer_phi)
            #loss = loss + F.cross_entropy(y_hat, y, weight=self.trainloader.dataset.classweights.to(self.args.device))
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            return loss

        #Gradient-Based Hyperparamter optimization algorithm.
        best_vauc = 0.0
        episodes_without_improvement = 0
        self.best_auc_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_auc_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())

        best_vbrier = float('inf')
        episodes_without_improvement = 0
        self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_brier_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
        
        best_vnll = float('inf')
        episodes_without_improvement = 0
        self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_nll_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())

        trainloader = InfIterator(self.trainloader)
        validloader = InfIterator(self.validloader)
       
        self.optimizermixer.train()
        for episode in tqdm(range(self.args.outer_episodes), ncols=75, leave=False):
            tlosses = []
            for k in tqdm(range(self.args.inner_episodes), ncols=75, leave=False):
                x, y = next(trainloader)
                context = self.contextloader[torch.randperm(len(self.contextloader))[0]]
                
                train_loss = train_mixer(model=self.model, optimizer=self.optimizer, mixer_phi=self.mixer_phi,\
                        x=x, y=y, context=context, device=self.args.device, interp_loss=True)
                tlosses.append(train_loss.item())
                
            #Compute hypergradients
            x_t, y_t = next(trainloader) #self.trainloader.dataset.get_batch(batch_size=50*self.args.train_batch_size)
            context_t = self.contextloader[torch.randperm(len(self.contextloader))[0]]
            L_T = train_mixer(model=self.model, optimizer=None, mixer_phi=self.mixer_phi, x=x_t, y=y_t, \
                    context=context_t, device=self.args.device)
        
            self.model.eval(); self.mixer_phi.eval()
            x_v, y_v = next(validloader) #self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
            context_v = None #self.contextloader[torch.randperm(len(self.contextloader))[0]].to(self.args.device)
            y_v_hat = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi)
            
            L_V = F.cross_entropy(y_v_hat[:, 1], y_v.to(self.args.device)) #, weight=self.validloader.dataset.classweights.to(self.args.device))
            
            hgrads = hypergradients(L_V=L_V, L_T=L_T, lmbda=self.mixer_phi.parameters, w=self.model.parameters, i=5, alpha=self.args.lr)
            
            self.optimizermixer.zero_grad()
            for p, g in zip(self.mixer_phi.parameters(), hgrads):
                hypergrad = torch.clamp(g, 5.0, 5.0)
                hypergrad *= 1.0 - (episode / (self.args.outer_episodes))
                p.grad = hypergrad
            self.optimizermixer.step()
            
            #Run model on validation set.
            vnll, vauc, vbrier = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)
            
            print('Episode: {:<3} tloss: {:.3f} vnll: {:.3f} vauc: {:.3f} vbrier: {:.3f}'.format(\
                        episode, np.mean(tlosses), vnll, vauc, vbrier))
            
            if vauc > best_vauc:
                best_vauc = vauc
                episodes_without_improvement = 0
                self.best_auc_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_auc_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
            else:
                episodes_without_improvement += 1
                if episodes_without_improvement == self.args.early_stopping_episodes:
                    break

            if vbrier < best_vbrier:
                best_vbrier = vbrier
                episodes_without_improvement = 0
                self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_brier_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
            
            if vnll < best_vnll:
                best_vnll = vnll
                episodes_without_improvement = 0
                self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_nll_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
         
    def fit(self):
        if self.mixer_phi is None:
            self.tlosses, self.vnlls, self.vaucs, self.vbriers, self.tnlls, self.taucs, self.tbriers = [], [], [], [], [], [], []
            
            best_vauc = 0.0
            self.best_auc_valid_state_dict_model = deepcopy(self.model.state_dict())

            best_vbrier = float('inf')
            self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
            
            best_vnll = float('inf')
            self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())

            for epoch in tqdm(range(self.args.epochs), total=self.args.epochs, ncols=75):
                tloss= self.train()
                vnll, vauc, vbrier = self.test(dataloader=self.validloader)
                print('Epoch: {:<3} tloss: {:.3f} vnll: {:.3f} vauc: {:.3f} vbrier: {:.3f}'.format(\
                        epoch, tloss, vnll, vauc, vbrier))
                self.tlosses.append(tloss)
                self.vnlls.append(vnll); self.vaucs.append(vauc), self.vbriers.append(vbrier)

                if vauc > best_vauc:
                    best_vauc = vauc
                    episodes_without_improvement = 0
                    self.best_auc_valid_state_dict_model = deepcopy(self.model.state_dict())
                else:
                    episodes_without_improvement += 1
                    if episodes_without_improvement == self.args.early_stopping_episodes:
                        break

                if vbrier < best_vbrier:
                    best_vbrier = vbrier
                    episodes_without_improvement = 0
                    self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
                
                if vnll < best_vnll:
                    best_vnll = vnll
                    episodes_without_improvement = 0
                    self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())
        else:
            self.train_mixer_phi()
        
        #Run model on test set.
        print('{} {} {}'.format(self.args.dataset, self.args.split_type, self.args.fingerprint))
        
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Last Model) Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))

        self.model.load_state_dict(self.best_auc_valid_state_dict_model)
        if self.mixer_phi is not None: 
            self.mixer_phi.load_state_dict(self.best_auc_valid_state_dict_mixer_phi)
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Best AUC)   Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))
        
        self.model.load_state_dict(self.best_brier_valid_state_dict_model)
        if self.mixer_phi is not None: 
            self.mixer_phi.load_state_dict(self.best_brier_valid_state_dict_mixer_phi)
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Best BRIER) Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))

        self.model.load_state_dict(self.best_nll_valid_state_dict_model)
        if self.mixer_phi is not None: 
            self.mixer_phi.load_state_dict(self.best_nll_valid_state_dict_mixer_phi) 
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Best NLL)   Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))

if __name__ == '__main__':
    args = get_arguments()

    #set_seed(args.seed)
        
    trainloader, validloader, testloader, contextloader = get_dataset(args=args)
    print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
    model, mixer_phi = get_model(args=args)
    
    optimizer = get_optimizer(optimizer=args.optimizer, model=model, lr=args.lr, wd=args.wd)
    optimizermixer = None if mixer_phi is None else get_optimizer(optimizer=args.optimizer, model=mixer_phi, lr=args.clr, wd=args.cwd)
    
    trainer = Trainer(model=model.to(args.device), \
            mixer_phi=mixer_phi if mixer_phi is None else mixer_phi.to(args.device), \
            optimizer=optimizer, \
            optimizermixer = optimizermixer, \
            trainloader=trainloader, \
            validloader=validloader, \
            contextloader=contextloader, \
            testloader=testloader, \
            args=args
            )

    trainer.fit()
