import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from utils import InfIterator
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

class Trainer:
    def __init__(self, epochs=500, \
            model=None, \
            contextmixer=None, \
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
        self.contextmixer = contextmixer
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

    def test(self, dataloader, contextmixer=None):
        self.model.eval()
        self.optimizer.eval()
        if contextmixer is not None:
            contextmixer.eval()
        
        with torch.no_grad():
            preds, labels = [], []
            for x, y in dataloader:
                y_hat = self.model(x=x.to(self.args.device), contextmixer=contextmixer)
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
    
    def train_contextmixer(self):
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
            
        def train_mixer(model, optimizer, contextmixer, x, y, context, device, interp_loss=False):
            model.train(); contextmixer.train()
            if optimizer is not None:
                optimizer.train()
            
            train_loss, train_acc = 0.0, 0.0

            x, y, context = x.to(device), y.to(device), context.to(device)
            
            #1. Mix context with labeled sample x
            y_hat_mixed = model(x=x, context=context, contextmixer=contextmixer)
            loss = F.cross_entropy(y_hat_mixed, y, weight=self.trainloader.dataset.classweights.to(self.args.device))
            
            #2. Pass unmixed sample through model
            y_hat = model(x=x, context=None, contextmixer=contextmixer)
            loss = loss + F.cross_entropy(y_hat, y, weight=self.trainloader.dataset.classweights.to(self.args.device))

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
        self.best_auc_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())

        best_vbrier = float('inf')
        episodes_without_improvement = 0
        self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_brier_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())
        
        best_vnll = float('inf')
        episodes_without_improvement = 0
        self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_nll_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())

        trainloader = InfIterator(self.trainloader)
        validloader = InfIterator(self.validloader)
       
        self.optimizermixer.train()
        for episode in tqdm(range(self.args.outer_episodes), ncols=75, leave=False):
            tlosses = []
            for k in tqdm(range(self.args.inner_episodes), ncols=75, leave=False):
                x, y = next(trainloader)
                context = self.contextloader[torch.randperm(len(self.contextloader))[0]]
                
                train_loss = train_mixer(model=self.model, optimizer=self.optimizer, contextmixer=self.contextmixer,\
                        x=x, y=y, context=context, device=self.args.device, interp_loss=True)
                tlosses.append(train_loss.item())
                
            #Compute hypergradients
            x_t, y_t = next(trainloader) #self.trainloader.dataset.get_batch(batch_size=50*self.args.train_batch_size)
            context_t = self.contextloader[torch.randperm(len(self.contextloader))[0]]
            L_T = train_mixer(model=self.model, optimizer=None, contextmixer=self.contextmixer, x=x_t, y=y_t, \
                    context=context_t, device=self.args.device)
        
            self.model.eval(); self.contextmixer.eval()
            x_v, y_v = next(validloader) #self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
            context_v = None #self.contextloader[torch.randperm(len(self.contextloader))[0]].to(self.args.device)
            y_v_hat = self.model(x=x_v.to(self.args.device), context=context_v, contextmixer=self.contextmixer)
            
            L_V = F.cross_entropy(y_v_hat[:, 1], y_v.to(self.args.device)) #, weight=self.validloader.dataset.classweights.to(self.args.device))
            
            hgrads = hypergradients(L_V=L_V, L_T=L_T, lmbda=self.contextmixer.parameters, w=self.model.parameters, i=5, alpha=self.args.lr)
            
            self.optimizermixer.zero_grad()
            for p, g in zip(self.contextmixer.parameters(), hgrads):
                hypergrad = torch.clamp(g, 5.0, 5.0)
                hypergrad *= 1.0 - (episode / (self.args.outer_episodes))
                p.grad = hypergrad
            self.optimizermixer.step()
            
            #Run model on validation set.
            vnll, vauc, vbrier = self.test(dataloader=self.validloader, contextmixer=self.contextmixer)
            
            print('Episode: {:<3} tloss: {:.3f} vnll: {:.3f} vauc: {:.3f} vbrier: {:.3f}'.format(\
                        episode, np.mean(tlosses), vnll, vauc, vbrier))
            
            if vauc > best_vauc:
                best_vauc = vauc
                episodes_without_improvement = 0
                self.best_auc_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_auc_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())
            else:
                episodes_without_improvement += 1
                if episodes_without_improvement == self.args.early_stopping_episodes:
                    break

            if vbrier < best_vbrier:
                best_vbrier = vbrier
                episodes_without_improvement = 0
                self.best_brier_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_brier_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())
            
            if vnll < best_vnll:
                best_vnll = vnll
                episodes_without_improvement = 0
                self.best_nll_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_nll_valid_state_dict_contextmixer = deepcopy(self.contextmixer.state_dict())
         
    def fit(self):
        if self.contextmixer is None:
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
            self.train_contextmixer()
        
        #Run model on test set.
        print('{} {} {}'.format(self.args.dataset, self.args.split_type, self.args.fingerprint))
        
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, contextmixer=self.contextmixer)
        print('(Last Model) Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))

        self.model.load_state_dict(self.best_auc_valid_state_dict_model)
        if self.contextmixer is not None: 
            self.contextmixer.load_state_dict(self.best_auc_valid_state_dict_contextmixer)
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, contextmixer=self.contextmixer)
        print('(Best AUC)   Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))
        
        self.model.load_state_dict(self.best_brier_valid_state_dict_model)
        if self.contextmixer is not None: 
            self.contextmixer.load_state_dict(self.best_brier_valid_state_dict_contextmixer)
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, contextmixer=self.contextmixer)
        print('(Best BRIER) Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))

        self.model.load_state_dict(self.best_nll_valid_state_dict_model)
        if self.contextmixer is not None: 
            self.contextmixer.load_state_dict(self.best_nll_valid_state_dict_contextmixer) 
        tnll, tauc, tbrier = self.test(dataloader=self.testloader, contextmixer=self.contextmixer)
        print('(Best NLL)   Tnll: {:.3f} Tauc: {:.3f} Tbrier: {:.3f}'.format(tnll, tauc, tbrier))
