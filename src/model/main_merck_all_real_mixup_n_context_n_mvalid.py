import os
import copy
import torch
import random
import numpy as np
import itertools
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle
from setenc import get_mixer
from arguments import get_arguments
from utils import set_seed, get_optimizer, InfIterator
from main_origin import get_model

# exclude_three = ['3a4', 'cb1', 'hivint', 'logd', 'metab', 'ox1', 'ox2', 'pgp', 'ppb', 'rat_f', 'tdi', 'thrombin']
exclude_three = ['dpp4', 'hivprot', 'nk1']
# contextloader don't include other 2

class Merck(Dataset):
    def __init__(self, split="train", vec_type='count', dataset="hivprot", is_context=False, batchsize=64, mvalid_dataset=None, exclude_mval_data_in_context=None, context_dataset=None):
        self.batchsize = batchsize
        self.vec_type = vec_type
        self.dataset = dataset
        self.split = split
        self.is_context = is_context
        self.mvalid_dataset = mvalid_dataset
        self.exclude_mval_data_in_context = exclude_mval_data_in_context
        self.context_dataset = context_dataset
        
        self.data, self.labels = self.load_data()
        self.mu = 0
        self.sigma = 1
        

    def load_data(self):
        files = os.listdir("/c2/jinakim/dataset_backup/Merck/Merck/preprocessed/")

        def filter(s):
            if self.is_context:
                
                if not self.exclude_mval_data_in_context:
                    if self.context_dataset != None:
                        return all(d not in s.lower() for d in exclude_three) and all(d not in s.lower() for d in self.context_dataset) and self.split in s and ".pt" in s
                    else:
                        return all(d not in s.lower() for d in exclude_three) and self.split in s and ".pt" in s
                        
                    # return self.dataset not in s.lower() and self.split in s and ".pt" in s
                else:
                    if self.context_dataset != None:
                        return (
                        # self.dataset not in s.lower()
                        all(d not in s.lower() for d in exclude_three)
                        and all(d not in s.lower() for d in self.context_dataset)
                        and self.split in s
                        and ".pt" in s
                        and all(md not in s.lower() for md in self.mvalid_dataset)
                        )
                    else:
                        return (
                            # self.dataset not in s.lower()
                            all(d not in s.lower() for d in exclude_three)
                            and self.split in s
                            and ".pt" in s
                            and all(md not in s.lower() for md in self.mvalid_dataset)
                        )

            return self.dataset in s.lower() and self.split in s and ".pt" in s

        files = [f for f in files if filter(f)]
        
        # print('=====')
        # print(f'MERCK_REAL exclude {self.exclude_mval_data_in_context}')
        # print('is_context ', self.is_context)
        # print('mvalid_dataset ', self.mvalid_dataset)
        # print('files ', files)
        # print('exc context ', self.context_dataset)
        
        data = [torch.load(f"/c2/jinakim/dataset_backup/Merck/Merck/preprocessed/{f}").float() for f in files]
        max_dim = 6561

        data = [torch.cat((d, torch.zeros(d.size(0), max_dim - d.size(1))), dim=1) for d in data]
        data = torch.cat(data, dim=0)
        data, labels = data[:, 1:], data[:, 0]

        data = torch.exp(data) - 1

        with open("/c2/jinakim/dataset_backup/Merck/Merck/preprocessed/stats.pkl", "rb") as f:
            stats = pickle.load(f)

        stats = [v for v in stats if v[0].lower() == self.dataset]
        mu, sigma = stats[0][1:3]
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        # print(f"{self.mu.item()=} {self.sigma.item()=}")

        return data, labels

    def denormalize(self, y):
        return y * self.sigma + self.mu

    def __getitem__(self, index):
        if self.vec_type == "count":
            return self.data[index], self.labels[index]
        elif self.vec_type == "bit":
            return (self.data[index] > 1e-2).float(), self.labels[index]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.data)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(args, test=False):
    
    if args.num_inner_dataset == 1:
        if args.same_setting:
            print(f"Inner {args.dataset=} {args.vec_type=}") # hivprot
            trainset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context) # hivprot

            if test:
                # if testing, just copy the trainset since we will take the last model no matter what
                validset = copy.deepcopy(trainset)
                perm = np.random.permutation(validset.data.shape[0])

                # make it small so we don't waste too much calculation
                n = int(perm.shape[0] * 0.1)
                idx = perm[:n]

                validset.data = validset.data[idx]
                validset.labels = validset.labels[idx]
            else:
                # if we are not testing, then we are tuning hyperparameters. In this case, we should
                # split the train set into a validation set for hyperparameter selection
                train_idx = np.load(f"data/perms/train-idx-{args.dataset}.npy")
                trainset.data = trainset.data[train_idx]
                trainset.labels = trainset.labels[train_idx]

                validset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
                val_idx = np.load(f"data/perms/val-idx-{args.dataset}.npy")
                validset.data = validset.data[val_idx]
                validset.labels = validset.labels[val_idx]

            testset = Merck(split="test", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context) # hivprot

            # validset for outer loop
            
            if os.environ.get('MIX_TYPE', 'SET') in ['MIXUP', 'MANIFOLD_MIXUP', 'SET_NO_BILEVEL']:
                # 'MIXUP_BILEVEL' 'MANIFOLD_MIXUP_BILEVEL' has mvalidset
                mvalidset = None
            else:
                if os.environ.get('MVALID_DEFAULT', '1') !=  '0':
                    if args.n_mvalid < 0:
                        mvalidset = None
                        # print('MVALID ', args.mvalid_dataset)
                        for dataset in args.mvalid_dataset:
                            _validset = Merck(split="train", vec_type=args.vec_type, dataset=dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context) # "dpp4", "nk1"
                            if mvalidset is None:
                                mvalidset = _validset
                                continue

                            mvalidset.data = torch.cat((mvalidset.data, _validset.data), dim=0)
                            mvalidset.labels = torch.cat((mvalidset.labels, _validset.labels), dim=0)
                    else:
                        mvalidset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=True, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context, context_dataset=args.context_dataset)
                else:
                    mvalidset = copy.deepcopy(validset)
        else:
            print(f"Inner {args.dataset=} {args.vec_type=}")
            trainset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)

            if test:
                # if testing, just copy the trainset since we will take the last model no matter what
                validset = copy.deepcopy(trainset)
                perm = np.random.permutation(validset.data.shape[0])

                # make it small so we don't waste too much calculation
                n = int(perm.shape[0] * 0.1)
                idx = perm[:n]

                validset.data = validset.data[idx]
                validset.labels = validset.labels[idx]
            else:
                # if we are not testing, then we are tuning hyperparameters. In this case, we should
                # split the train set into a validation set for hyperparameter selection
                train_idx = np.load(f"data/perms/train-idx-{args.dataset}.npy")
                trainset.data = trainset.data[train_idx]
                trainset.labels = trainset.labels[train_idx]

                validset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
                val_idx = np.load(f"data/perms/val-idx-{args.dataset}.npy")
                validset.data = validset.data[val_idx]
                validset.labels = validset.labels[val_idx]

            testset = Merck(split="test", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)

            # validset for outer loop
            mvalidset = None
            for dataset in [d for d in ["hivprot", "dpp4", "nk1"] if d != args.dataset]:
                _validset = Merck(split="train", vec_type=args.vec_type, dataset=dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
                if mvalidset is None:
                    mvalidset = _validset
                    continue

                mvalidset.data = torch.cat((mvalidset.data, _validset.data), dim=0)
                mvalidset.labels = torch.cat((mvalidset.labels, _validset.labels), dim=0)
    elif args.num_inner_dataset == 2:
        raise NotImplementedError()
        # TODO implement not finished yet
        print(f"Outer {args.dataset=} {args.vec_type=}")
        # validset for outer loop
        trainset = None
        for dataset in [d for d in ["hivprot", "dpp4", "nk1"] if d != args.dataset]:
            _trainset = Merck(split="train", vec_type=args.vec_type, dataset=dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
            if trainset is None:
                trainset = _trainset
                continue

            trainset.data = torch.cat((trainset.data, _trainset.data), dim=0)
            trainset.labels = torch.cat((trainset.labels, _trainset.labels), dim=0)
        
        mvalidset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)

        if test:
            # if testing, just copy the trainset since we will take the last model no matter what
            validset = copy.deepcopy(trainset)
            perm = np.random.permutation(validset.data.shape[0])

            # make it small so we don't waste too much calculation
            n = int(perm.shape[0] * 0.1)
            idx = perm[:n]

            validset.data = validset.data[idx]
            validset.labels = validset.labels[idx]
        else:
            # if we are not testing, then we are tuning hyperparameters. In this case, we should
            # split the train set into a validation set for hyperparameter selection
            train_idx = np.load(f"data/perms/train-idx-{args.dataset}.npy")
            trainset.data = trainset.data[train_idx]
            trainset.labels = trainset.labels[train_idx]

            validset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
            val_idx = np.load(f"data/perms/val-idx-{args.dataset}.npy")
            validset.data = validset.data[val_idx]
            validset.labels = validset.labels[val_idx]

        testset = Merck(split="test", vec_type=args.vec_type, dataset=args.dataset, is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)

        
    else:
        raise NotImplementedError()

    m = trainset.data.amax()
    trainset.data = trainset.data / m
    validset.data = validset.data / m
    
    if mvalidset is not None:
        mvalidset.data = mvalidset.data / m
    testset.data = testset.data / m
    
    # print(f"{trainset.data.shape=} {trainset.labels.shape=}")
    # print(f"{validset.data.shape=} {validset.labels.shape=}")
    # print(f"{mvalidset.data.shape=} {mvalidset.labels.shape=}")
    # print(f"{testset.data.shape=} {testset.labels.shape=}")
    
    
    ood1_trainloader = None
    ood2_trainloader = None
    
    if args.tsne_plot:
        ood1_trainset = Merck(split="train", vec_type=args.vec_type, dataset=args.specify_ood_dataset[0], is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
        ood2_trainset = Merck(split="train", vec_type=args.vec_type, dataset=args.specify_ood_dataset[1], is_context=False, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context)
        ood1_trainset.data = ood1_trainset.data/ m
        ood2_trainset.data = ood2_trainset.data/ m
        
        g_ood1 = torch.Generator()
        g_ood1.manual_seed(0) # NOTE JIN : we're only using seed 0 for tsne
        
        g_ood2 = torch.Generator()
        g_ood2.manual_seed(0)
        
        ood1_trainloader = DataLoader(
            ood1_trainset,
            drop_last=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g_ood1,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True
        )
        
        ood2_trainloader = DataLoader(
            ood2_trainset,
            drop_last=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g_ood2,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True
        )
        
        g_train = torch.Generator()
        g_train.manual_seed(0) # NOTE JIN : we're only using seed 0 for tsne
        
        g_context = torch.Generator()
        g_context.manual_seed(0)
        
        g_mvalid = torch.Generator()
        g_mvalid.manual_seed(0)
    else:
        g_train = None
        g_mvalid = None
        g_context = None
        

    # print(f"{trainset.data.shape=} {trainset.labels.shape=}")
    # print(f"{validset.data.shape=} {validset.labels.shape=}")
    # print(f"{mvalidset.data.shape=} {mvalidset.labels.shape=}")
    # print(f"{testset.data.shape=} {testset.labels.shape=}")

    
    # g.manual_seed(0)

    # if args.tsne_plot:
    #     train_indices = torch.randperm(len(trainset))
    #     torch.save(train_indices, "train_shuffle_indices.pt")
    #     train_subset = torch.utils.data.Subset(trainset, train_indices)
        
    #     trainloader = DataLoader(
    #         train_subset,
    #         drop_last=True,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         worker_init_fn=seed_worker,
    #         # generator=g,
    #         persistent_workers=True,
    #         shuffle=False,
    #         pin_memory=True
    #     )
        
    #     contextloader = None
    #     if args.mixer_phi:
    #         #  14 'DPP4_train.pt', 'OX2_train.pt', '3A4_train.pt', 'OX1_train.pt', 'PPB_train.pt', 'CB1_train.pt', 'HIVINT_train.pt', 'THROMBIN_train.pt', 'PGP_train.pt', 'NK1_train.pt', 'RAT_F_train.pt', 'TDI_train.pt', 'METAB_train.pt', 'LOGD_train.pt'
    #         contextset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=True, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context, context_dataset=args.context_dataset)
    #         contextset.data = contextset.data / m
            
    #         context_indices = torch.randperm(len(contextset))
    #         torch.save(context_indices, "context_shuffle_indices.pt")
    #         context_subset = torch.utils.data.Subset(contextset, context_indices)
            
    #         contextloader = DataLoader(
    #             context_subset,
    #             batch_size=args.batch_size * args.n_context,
    #             drop_last=True,
    #             num_workers=args.num_workers,
    #             persistent_workers=True,
    #             worker_init_fn=seed_worker,
    #                 # generator=g,
    #                 shuffle=False,
    #                 pin_memory=True
    #         )
    # else:
    trainloader = DataLoader(
        trainset,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g_train,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True
    )
    
    contextloader = None
    if args.mixer_phi:
        #  14 'DPP4_train.pt', 'OX2_train.pt', '3A4_train.pt', 'OX1_train.pt', 'PPB_train.pt', 'CB1_train.pt', 'HIVINT_train.pt', 'THROMBIN_train.pt', 'PGP_train.pt', 'NK1_train.pt', 'RAT_F_train.pt', 'TDI_train.pt', 'METAB_train.pt', 'LOGD_train.pt'
        contextset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=True, mvalid_dataset=args.mvalid_dataset, exclude_mval_data_in_context=args.exclude_mval_data_in_context, context_dataset=args.context_dataset)
        contextset.data = contextset.data / m
        contextloader = DataLoader(
            contextset,
            batch_size=args.batch_size * args.n_context,
            drop_last=True,
            num_workers=args.num_workers,
            persistent_workers=True,
            worker_init_fn=seed_worker,
                generator=g_context,
                shuffle=True,
                pin_memory=True
        )
    validloader = DataLoader(
        validset,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        # generator=g,
        shuffle=False,
        pin_memory=True
    )

    if mvalidset is not None:
        if args.n_mvalid < 0:
            mvalidloader = DataLoader(
                mvalidset,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                generator=g_mvalid,
                shuffle=True,
                pin_memory=True
            )
        else:
            mvalidloader = DataLoader(
                mvalidset,
                batch_size=args.batch_size * args.n_mvalid,
                drop_last=True,
                num_workers=args.num_workers,
                persistent_workers=True,
                worker_init_fn=seed_worker,
                    generator=g_mvalid,
                    shuffle=True,
                    pin_memory=True
            )
    else:
        mvalidloader = None

    testloader = DataLoader(testset, \
                            batch_size=args.batch_size, \
                            num_workers=args.num_workers, \
                            persistent_workers=True,
                            worker_init_fn=seed_worker, \
                            # generator=g,
                            shuffle=False, pin_memory=True)

        
    return trainloader, validloader, mvalidloader, testloader, contextloader, ood1_trainloader, ood2_trainloader


class Trainer:
    def __init__(
        self, epochs=500,
        model=None,
        mixer_phi=None,
        optimizer=None,
        testloader=None,
        trainloader=None,
        validloader=None,
        mvalidloader=None,
        contextloader=None,
        ood1_trainloader=None, 
        ood2_trainloader=None, 
        optimizermixer=None,
        args=None,
        seed=0
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.testloader = testloader
        self.trainloader = trainloader
        self.validloader = validloader
        self.mvalidloader = mvalidloader
        self.mixer_phi = mixer_phi
        self.contextloader = contextloader
        self.ood1_trainloader = ood1_trainloader
        self.ood2_trainloader = ood2_trainloader
        self.optimizermixer = optimizermixer
        self.embedding_list = []
        self.label_list = []
        self.loss_list = []

    def train(self):
        losses = []
        self.model.train()
        self.optimizer.train()
        
        for x, y in self.trainloader:
            y_hat = self.model(x=x.to(self.args.device))
            loss = F.mse_loss(y_hat[0].squeeze(), y.to(self.args.device))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)
    

    def calc_loss(self, y_hat, y, test=False):
        if len(y_hat.size()) > 1:

            raise ValueError(f"sizes should not happen: {y_hat.size()=} {y.size()=}")
            y_hat_real = y_hat[:, 0]
            loss = F.mse_loss(y, y_hat_real)

            y_hat_unlabeled = y_hat[:, 1:].reshape(-1)
            mu = y_hat_unlabeled.mean()
            var = y_hat_unlabeled.var()
            logvar = var.log()

            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
            # print(f"{loss=} {kl=}")
            loss += 0.01 * kl
            return loss

        # print(f"regular loss: {y.size()=} {y_hat.size()=}")
        return F.mse_loss(y.cuda().squeeze(), y_hat.cuda().squeeze())

    def test(self, dataloader, contextloader=None, mixer_phi=None, embed_type=None):
        self.model.eval()
        self.optimizer.eval() # why????
        if mixer_phi is not None:
            mixer_phi.eval()
        
        if self.args.tsne_plot and embed_type == "train_context_bunch":
            raise Exception()
            with torch.no_grad():
                losses = []
                counts = 0
                for episode in tqdm(range(self.args.outer_episodes * self.args.inner_episodes), ncols=75, leave=False):
                    x, y = next(dataloader)
                    context, context_y = next(contextloader)
                    
                    context = context.reshape(self.args.batch_size, -1, context.size(-1))
                    if args.n_context > 1:
                        n = torch.randint(1, context.size(1), size=(1,)).item()
                        context = context[:, :n]
                        
                        # context_y = context_y.reshape(self.args.batch_size, -1)
                        # context_y = context_y[:, :n].reshape(-1)
                
                    x, y, context = x.to(self.args.device), y.to(self.args.device), context.to(self.args.device)
                    
                    # if self.args.embed_test == "base_cX_mO": # NOTE context=context
                    #     #### TODO check first iteration pairs
                    # if episode == 0 and embed_type == "train_context":
                    #     print('====')
                    #     print('x ', x)
                    #     print('====')
                    #     print('context ', context)
                    #     print('====')
                    #     print('x ', x.shape)
                    #     print('context ', context.shape)
                    #     torch.save(x, 'test_x.pt')
                    #     torch.save(context, 'test_context.pt')
                            
                    # breakpoint()#### TODO check first iteration pairs
                    #         torch.save(context_y, 'test_context_y_train_context.pt') # TODO want this to be same with above
                            
                    #     B, S, H = context.size()
                    #     context = context.view(B*S, H)
                        
                    #     x = torch.cat([x, context], dim=0)
                    #     y = torch.cat([y, context_y], dim=0)
                        # context = None
                        
                    y_hat, self.embedding_list, self.label_list = self.model(x=x.to(self.args.device), context=context, mixer_phi=mixer_phi, embedding_list=self.embedding_list, label_list=self.label_list, embed_type=embed_type, embed_test=self.args.embed_test)
                    
                    y = y.cuda().squeeze()
                    y_hat = y_hat.cuda().squeeze()

                    # y_hat = y_hat[:, 0]
                    # print(f"in test: {y.size()=} {y_hat.size()=}")

                    loss = self.calc_loss(y_hat, y, test=True)
                    self.loss_list.append(torch.full((x.shape[0],), loss.detach().item()))
                    
                    # breakpoint()
                    
                    losses.append(loss.item() * y_hat.size(0))
                    counts += y_hat.size(0)
                    
            mse = sum(losses) / counts
            return mse
            
        elif self.args.tsne_plot and embed_type == "train_context":
            raise Exception()
            with torch.no_grad():
                losses = []
                counts = 0
                for episode in tqdm(range(self.args.outer_episodes * self.args.inner_episodes), ncols=75, leave=False):
                    x, y = next(dataloader)
                    context, context_y = next(contextloader)
                    
                    context = context.reshape(self.args.batch_size, -1, context.size(-1))
                    if args.n_context > 1:
                        n = torch.randint(1, context.size(1), size=(1,)).item()
                        context = context[:, :n]
                        
                        # context_y = context_y.reshape(self.args.batch_size, -1)
                        # context_y = context_y[:, :n].reshape(-1)
                
                    x, y, context = x.to(self.args.device), y.to(self.args.device), context.to(self.args.device)
                    
                    # if self.args.embed_test == "base_cX_mO": # NOTE context=context
                    #     #### TODO check first iteration pairs
                    # if episode == 0 and embed_type == "train_context":
                    #     print('====')
                    #     print('x ', x)
                    #     print('====')
                    #     print('context ', context)
                    #     print('====')
                    #     print('x ', x.shape)
                    #     print('context ', context.shape)
                    #     torch.save(x, 'test_x.pt')
                    #     torch.save(context, 'test_context.pt')
                            
                    # breakpoint()#### TODO check first iteration pairs
                    #         torch.save(context_y, 'test_context_y_train_context.pt') # TODO want this to be same with above
                            
                    #     B, S, H = context.size()
                    #     context = context.view(B*S, H)
                        
                    #     x = torch.cat([x, context], dim=0)
                    #     y = torch.cat([y, context_y], dim=0)
                        # context = None
                        
                    y_hat, self.embedding_list, self.label_list = self.model(x=x.to(self.args.device), context=context, mixer_phi=mixer_phi, embedding_list=self.embedding_list, label_list=self.label_list, embed_type=embed_type, embed_test=self.args.embed_test)
                    
                    y = y.cuda().squeeze()
                    y_hat = y_hat.cuda().squeeze()

                    # y_hat = y_hat[:, 0]
                    # print(f"in test: {y.size()=} {y_hat.size()=}")

                    loss = self.calc_loss(y_hat, y, test=True)
                    self.loss_list.append(torch.full((x.shape[0],), loss.detach().item()))
                    
                    # breakpoint()
                    
                    losses.append(loss.item() * y_hat.size(0))
                    counts += y_hat.size(0)
                    
            mse = sum(losses) / counts
            return mse
        # elif self.args.tsne_plot and ("cX_mX" in self.args.embed_test) and embed_type == "mvalid_none":
        #     with torch.no_grad():
        #         losses = []
        #         counts = 0
        #         for episode in tqdm(range(self.args.outer_episodes), ncols=75, leave=False):
        #             x, y = next(dataloader)
                    
        #             x, y = x.to(self.args.device), y.to(self.args.device)
                    
        #             if episode == 0:
        #                 print('====')
        #                 print('x_v ', x)
        #                 print('====')
        #                 print('y_v ', y)

        #                 torch.save(x, f'test_x_v.pt')
        #                 torch.save(y, f'test_y_v.pt')
                        
        #             y_hat, self.embedding_list, self.label_list = self.model(x=x.to(self.args.device), context=context, mixer_phi=mixer_phi, embedding_list=self.embedding_list, label_list=self.label_list, embed_type=embed_type, embed_test=self.args.embed_test)
                    
        #             y = y.cuda().squeeze()
        #             y_hat = y_hat.cuda().squeeze()

        #             # y_hat = y_hat[:, 0]
        #             # print(f"in test: {y.size()=} {y_hat.size()=}")

        #             loss = self.calc_loss(y_hat, y, test=True)
        #             self.loss_list.append(torch.full((x.shape[0],), loss.detach().item()))
                    
        #             # breakpoint()
                    
        #             losses.append(loss.item() * y_hat.size(0))
        #             counts += y_hat.size(0)
                    
        #     mse = sum(losses) / counts
        #     return mse
        else:
            with torch.no_grad():
                losses = []
                counts = 0
                for x, y in dataloader:
                    y_hat, self.embedding_list, self.label_list = self.model(x=x.to(self.args.device), mixer_phi=mixer_phi, embedding_list=None, label_list=None, embed_type=None, embed_test=None)

                    y = y.cuda().squeeze()
                    y_hat = y_hat.cuda().squeeze()

                    # y_hat = y_hat[:, 0]
                    # print(f"in test: {y.size()=} {y_hat.size()=}")

                    loss = self.calc_loss(y_hat, y, test=True)
                    if self.args.tsne_plot:
                        self.loss_list.append(torch.full((x.shape[0],), loss.detach().item()))

                    losses.append(loss.item() * y_hat.size(0))
                    counts += y_hat.size(0)
            # self.model.eval()

            mse = sum(losses) / counts
            return mse

    def train_mixer_phi(self):
        def approxInverseHVP(v, f, w, i=5, alpha=0.1):
            p = [v_.clone().detach() for v_ in v]
            for j in range(i):
                grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)
                v = [v_ - alpha * g_ for v_, g_ in zip(v, grad)]
                p = [v_ + p_ for v_, p_ in zip(v, p)]
            return [alpha * p_ for p_ in p]

        def hypergradients(L_V, L_T, lmbda, w, i=5, alpha=0.1):
            v1 = torch.autograd.grad(L_V, w(), retain_graph=True)

            d_LT_dw = torch.autograd.grad(L_T, w(), create_graph=True)

            v2 = approxInverseHVP(v=v1, f=d_LT_dw, w=w, i=i, alpha=alpha)

            # for p in lmbda():
            #     if p.grad is None:
            #         print("❗ Unused parameter:", p.shape)
            
            v3 = torch.autograd.grad(d_LT_dw, lmbda(), grad_outputs=v2, retain_graph=True, allow_unused=True)
            
            # for i, (param, grad) in enumerate(zip(lmbda(), v3)):
            #     if grad is None:
            #         print(f"❗ Unused parameter #{i}: shape={param.shape}, name={param.__class__.__name__}")
            
            # for name, param in self.mixer_phi.named_parameters():
            #     if param.grad is None:
            #         print(f"❗ Parameter unused in L_T graph: {name} | shape={param.shape}")

                    # print(f"❗ Parameter unused in L_T graph: {name} | shape={param.shape}")
            d_LV_dlmbda = torch.autograd.grad(L_V, lmbda(), allow_unused=True)
            
            hgrads = []
            for d, v, p in zip(d_LV_dlmbda, v3, lmbda()):
                if d is None and v is None:
                    hgrads.append(torch.zeros_like(p))
                elif d is None:
                    hgrads.append(-v)
                elif v is None:
                    hgrads.append(d)
                else:
                    hgrads.append(d - v)
            
            return hgrads
            
            # return [d - v for d, v in zip(d_LV_dlmbda, v3)]

        def train_mixer(model, optimizer, mixer_phi, x, y, context, context_y, device, interp_loss=False):
            model.train()
            mixer_phi.train()
            if optimizer is not None:
                optimizer.train()

            x, y, context = x.to(device), y.to(device), context.to(device)

            # 1. Mix context with labeled sample x
            if self.args.model_no_context:
                context=None
            
            if self.args.tsne_plot and "base_cX_mO" in self.args.embed_test:
                # print('>> base_cX_mO // y ', y.shape)
                # print('>> base_cX_mO // context_y ', context_y.shape)
                # print('#########')
                # breakpoint()
                
                # context_y = context_y.to(device)
                # B, S, H = context.size()
                # context = context.view(B*S, H)
                
                # x = torch.cat([x, context], dim=0)
                # y = torch.cat([y, context_y], dim=0)
                raise Exception()
                context = None
            y_hat_mixed, _, _ = model(x=x, context=context, mixer_phi=mixer_phi, embedding_list=None, label_list=None, embed_type=None)
            loss = self.calc_loss(y_hat_mixed.squeeze(), y.squeeze())

            # 2. Pass unmixed sample through model
            # y_hat = model(x=x, context=None, mixer_phi=mixer_phi)
            # loss = loss + self.calc_loss(y_hat.squeeze(), y.squeeze())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            return loss

        # Gradient-Based Hyperparamter optimization algorithm.
        best_vmse = float("inf")
        episodes_without_improvement = 0
        self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())

        trainloader = InfIterator(self.trainloader)
        mvalidloader = InfIterator(self.mvalidloader)
        contextloader = InfIterator(self.contextloader)

        assert os.environ.get('MIX_TYPE', 'SET') in ['MIXUP_BILEVEL', 'MANIFOLD_MIXUP_BILEVEL']
        if os.environ.get('MIX_TYPE', 'SET') == 'MANIFOLD_MIXUP_BILEVEL':
            self.optimizermixer.train()
        
        for episode in tqdm(range(self.args.outer_episodes), ncols=75, leave=False):
            tlosses = []
            for k in tqdm(range(self.args.inner_episodes), ncols=75, leave=False):
                x, y = next(trainloader)
                context, context_y = next(contextloader)
                context = context.reshape(self.args.batch_size, -1, context.size(-1))
                
                # print('context ', context.shape )
                # print('context_y ', context_y.shape )
                if args.n_context > 1:
                    n = torch.randint(1, context.size(1), size=(1,)).item()
                    context = context[:, :n]
                    
                    # if args.tsne_plot and args.embed_test == "base_cX_mO":
                    #     # context_y = context_y.reshape(self.args.batch_size, -1)
                    #     # context_y = context_y[:, :n].reshape(-1)
                        
                # if k == 0 and episode == 0:
                #     print('====')
                #     print('x ', x)
                #     print('====')
                #     print('context ', context)
                #     print('====')
                #     print('x ', x.shape)
                #     print('context ', context.shape)

                #     torch.save(x, f'train_mixer_x.pt')
                #     torch.save(context, f'train_mixer_context.pt')
                            # torch.save(context_y, f'train_mixer_context_y.pt')
                        
                    # print('new context ', context.shape)
                    # print('new y ', context_y.shape)

                train_loss = train_mixer(model=self.model, optimizer=self.optimizer, mixer_phi=self.mixer_phi, \
                                         x=x, y=y, context=context, context_y=context_y, device=self.args.device, interp_loss=True)
                tlosses.append(train_loss.item())

            if os.environ.get('MIX_TYPE', 'SET') == 'MANIFOLD_MIXUP_BILEVEL':
                # Compute hypergradients
                x_t, y_t = next(trainloader)  # self.trainloader.dataset.get_batch(batch_size=50*self.args.train_batch_size)
                context_t, context_y_t = next(contextloader)
                context_t = context_t.reshape(self.args.batch_size, -1, context_t.size(-1))
                if args.n_context > 1:
                    n = torch.randint(1, context_t.size(1), size=(1,)).item()
                    context_t = context_t[:, :n]
                    
                    # if args.tsne_plot and args.embed_test == "base_cX_mO":
                    #     context_y_t = context_y_t.reshape(self.args.batch_size, -1)
                    #     context_y_t = context_y_t[:, :n].reshape(-1)

                L_T = train_mixer(model=self.model, optimizer=None, mixer_phi=self.mixer_phi, x=x_t, y=y_t, \
                                context=context_t, context_y=context_y_t, device=self.args.device)

                # self.model.eval()
                # self.mixer_phi.eval()
                x_v, y_v = next(mvalidloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
                
                if os.environ.get('RANDOM_YV', '0') =='1':
                    # print('y_v ', y_v.shape)
                    # breakpoint()
                    y_v = torch.randn_like(y_v)
                
                # x_v, y_v = next(trainloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
                
                # if episode == 0:
                #     print('====')
                #     print('x_v ', x_v)
                #     print('====')
                #     print('y_v ', y_v)

                #     torch.save(x_v, f'train_ours_x_v.pt')
                #     torch.save(y_v, f'train_ours_y_v.pt')
                
                context_v = None
                if args.tsne_plot and episode == self.args.outer_episodes - 1:
                    raise Exception()
                    y_v_hat, self.embedding_list, self.label_list = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi, embedding_list=self.embedding_list, label_list=self.label_list, embed_type="mvalid_none", embed_test=self.args.embed_test)
                    L_V = self.calc_loss(y_v_hat.squeeze(), y_v.to(self.args.device).squeeze(), test=False)  # , weight=self.validloader.dataset.classweights.to(self.args.device))
                    self.loss_list.append(torch.full((x_v.shape[0],), L_V.detach().item()))
                    
                else:
                    y_v_hat, _, __ = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                    
                    # y_v_hat = y_v_hat[:, 0]
                    L_V = self.calc_loss(y_v_hat.squeeze(), y_v.to(self.args.device).squeeze(), test=False)  # , weight=self.validloader.dataset.classweights.to(self.args.device))
                
                def w():
                    return self.model.parameters()

                hgrads = hypergradients(L_V=L_V, L_T=L_T, lmbda=self.mixer_phi.parameters, w=w, i=5, alpha=self.args.lr)

                self.optimizermixer.zero_grad()
                for p, g in zip(self.mixer_phi.parameters(), hgrads):
                    hypergrad = torch.clamp(g, -5.0, 5.0)
                    # hypergrad *= 1.0 - (episode / (self.args.outer_episodes))
                    p.grad = hypergrad
                self.optimizermixer.step()

            elif os.environ.get('MIX_TYPE', 'SET') == 'MIXUP_BILEVEL':
                # Compute hypergradients
                # x_t, y_t = next(trainloader)  # self.trainloader.dataset.get_batch(batch_size=50*self.args.train_batch_size)
                # context_t, context_y_t = next(contextloader)
                # context_t = context_t.reshape(self.args.batch_size, -1, context_t.size(-1))
                # if args.n_context > 1:
                #     n = torch.randint(1, context_t.size(1), size=(1,)).item()
                #     context_t = context_t[:, :n]
                    
                #     # if args.tsne_plot and args.embed_test == "base_cX_mO":
                #     #     context_y_t = context_y_t.reshape(self.args.batch_size, -1)
                #     #     context_y_t = context_y_t[:, :n].reshape(-1)

                # L_T = train_mixer(model=self.model, optimizer=None, mixer_phi=self.mixer_phi, x=x_t, y=y_t, \
                #                 context=context_t, context_y=context_y_t, device=self.args.device)

                # self.model.eval()
                # self.mixer_phi.eval()
                x_v, y_v = next(mvalidloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
                
                if os.environ.get('RANDOM_YV', '0') =='1':
                    # print('y_v ', y_v.shape)
                    # breakpoint()
                    y_v = torch.randn_like(y_v)
                
                # x_v, y_v = next(trainloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
                
                # if episode == 0:
                #     print('====')
                #     print('x_v ', x_v)
                #     print('====')
                #     print('y_v ', y_v)

                #     torch.save(x_v, f'train_ours_x_v.pt')
                #     torch.save(y_v, f'train_ours_y_v.pt')
                
                context_v = None
                if args.tsne_plot and episode == self.args.outer_episodes - 1:
                    raise Exception()
                    y_v_hat, self.embedding_list, self.label_list = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi, embedding_list=self.embedding_list, label_list=self.label_list, embed_type="mvalid_none", embed_test=self.args.embed_test)
                    L_V = self.calc_loss(y_v_hat.squeeze(), y_v.to(self.args.device).squeeze(), test=False)  # , weight=self.validloader.dataset.classweights.to(self.args.device))
                    self.loss_list.append(torch.full((x_v.shape[0],), L_V.detach().item()))
                    
                else:
                    y_v_hat, _, __ = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                    
                    # y_v_hat = y_v_hat[:, 0]
                    L_V = self.calc_loss(y_v_hat.squeeze(), y_v.to(self.args.device).squeeze(), test=False)  # , weight=self.validloader.dataset.classweights.to(self.args.device))
                
                def w():
                    return self.model.parameters()

                if self.optimizer is not None:
                    self.optimizer.train()

                    self.optimizer.zero_grad()
                    L_V.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # hgrads = hypergradients(L_V=L_V, L_T=L_T, lmbda=self.mixer_phi.parameters, w=w, i=5, alpha=self.args.lr)

                # self.optimizermixer.zero_grad()
                # for p, g in zip(self.mixer_phi.parameters(), hgrads):
                #     hypergrad = torch.clamp(g, -5.0, 5.0)
                #     # hypergrad *= 1.0 - (episode / (self.args.outer_episodes))
                #     p.grad = hypergrad
                # self.optimizermixer.step()
            else:
                raise Exception()
            
            # Run model on validation set.
            vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)

            print('Episode: {:<3} tloss: {:.3f} vmse: {:.3f}'.format(\
                episode, np.mean(tlosses), vmse))

            if vmse < best_vmse:
                best_vmse = vmse
                # episodes_without_improvement = 0
                self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
            # else:
            #     episodes_without_improvement += 1
            #     # if episodes_without_improvement >= self.args.early_stopping_episodes and episode > int(self.args.outer_episodes * 0.1):
            #     if episodes_without_improvement >= self.args.early_stopping_episodes and episode > args.outer_episodes:
            #         break
            
    def train_mixer_no_bilevel(self):
        # Gradient-Based Hyperparamter optimization algorithm.
        best_vmse = float("inf")
        episodes_without_improvement = 0
        self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())

        trainloader = InfIterator(self.trainloader)
        # mvalidloader = InfIterator(self.mvalidloader)
        contextloader = InfIterator(self.contextloader)
        

        # TODO NOTE # iteration could change 
        for episode in tqdm(range(int(os.environ.get('MIXER_NO_BILEVEL_EPOCHS', 10))), ncols=75, leave=False):
            tlosses = []

            self.model.train()
            self.mixer_phi.train()
            
            self.optimizer.train()
            
            for x, y in self.trainloader:
                context, context_y = next(contextloader)
                context = context.reshape(self.args.batch_size, -1, context.size(-1))
                
                if args.n_context > 1:
                    n = torch.randint(1, context.size(1), size=(1,)).item()
                    context = context[:, :n]

                x, y, context = x.to(self.args.device), y.to(self.args.device), context.to(self.args.device)
                
                if self.args.model_no_context:
                    raise Exception() # TODO implement for tsne
                    context=None

                y_hat, _, _ = self.model(x=x, context=None, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                
                    
                loss_model = self.calc_loss(y_hat.squeeze(), y.squeeze()) # TODO why y squeeze?
                
                loss_mixer = None
                if os.environ.get('SET_NO_BILEVEL_MIX_LABEL', 'true_y') == 'true_y':
                    y_hat_mixed, _, _ = self.model(x=x, context=context, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                    loss_mixer = self.calc_loss(y_hat_mixed.squeeze(), y.squeeze())
                    
                elif os.environ.get('SET_NO_BILEVEL_MIX_LABEL', 'random') == 'random':
                    y_hat_mixed, _, _ = self.model(x=x, context=context, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                    loss_mixer = self.calc_loss(y_hat_mixed.squeeze(), torch.randn_like(y).squeeze())
                else:
                    raise Exception()
                
                total_loss = loss_model + loss_mixer
                
                tlosses.append(total_loss.item())

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    all_params = list(self.model.parameters()) + list(self.mixer_phi.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    self.optimizer.step()
                
            vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)

            print('Episode: {:<3} tloss: {:.3f} vmse: {:.3f}'.format(\
                episode, np.mean(tlosses), vmse))

            if vmse < best_vmse:
                best_vmse = vmse
                # episodes_without_improvement = 0
                self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
        

    def train_mixup(self):
        best_vmse = float("inf")
        episodes_without_improvement = 0
        self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
        self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
    
        trainloader = InfIterator(self.trainloader)
        contextloader = InfIterator(self.contextloader)
        
        
        # TODO NOTE # iteration could change 
        for episode in tqdm(range(int(os.environ.get('MIXUP_EPOCHS', 300))), ncols=75, leave=False):
            tlosses = []

            self.model.train()
            self.optimizer.train()
            self.mixer_phi.train()
            
            for x, y in self.trainloader:
                context, context_y = next(contextloader)
                context = context.reshape(self.args.batch_size, -1, context.size(-1))
                
                if args.n_context > 1:
                    n = torch.randint(1, context.size(1), size=(1,)).item()
                    context = context[:, :n]

                x, y, context = x.to(self.args.device), y.to(self.args.device), context.to(self.args.device)
                
                if self.args.model_no_context:
                    raise Exception() # TODO implement for tsne
                    context=None

                y_hat_mixed, _, _ = model(x=x, context=context, mixer_phi=self.mixer_phi, embedding_list=None, label_list=None, embed_type=None)
                loss = self.calc_loss(y_hat_mixed.squeeze(), y.squeeze()) # TODO why y squeeze?
                tlosses.append(loss.item())

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
            vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)

            print('Episode: {:<3} tloss: {:.3f} vmse: {:.3f}'.format(\
                episode, np.mean(tlosses), vmse))

            if vmse < best_vmse:
                best_vmse = vmse
                # episodes_without_improvement = 0
                self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
        


    def fit(self):
        # print('### mixer_phi ', self.mixer_phi)
        if self.mixer_phi is None:
            self.tlosses = []

            best_vmse = float("inf")
            self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())

            for epoch in tqdm(range(self.args.epochs), total=self.args.epochs, ncols=75):
                tloss = self.train()
                vmse = self.test(dataloader=self.validloader)
                print('Epoch: {:<3} tloss: {:.3f} vmse: {:.3f}'.format(epoch, tloss, vmse))
                self.tlosses.append(tloss)

                if vmse < best_vmse:
                    best_vmse = vmse
                    episodes_without_improvement = 0
                    self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
                else:
                    episodes_without_improvement += 1
                    if episodes_without_improvement == self.args.early_stopping_episodes:
                        break
        else:
            if os.environ.get('MIX_TYPE', 'SET') == 'MIXUP' or os.environ.get('MIX_TYPE', 'SET') == 'MANIFOLD_MIXUP':
                self.train_mixup()
            elif os.environ.get('MIX_TYPE', 'SET') == 'MIXUP_BILEVEL' or os.environ.get('MIX_TYPE', 'SET') == 'MANIFOLD_MIXUP_BILEVEL':
                self.train_mixer_phi()
            elif os.environ.get('MIX_TYPE', 'SET') == 'SET':
                self.train_mixer_phi()
            elif os.environ.get('MIX_TYPE', 'SET') == 'SET_NO_BILEVEL':
                self.train_mixer_no_bilevel()
            else:
                raise Exception()
            
            # self.train_mixer_phi()

        # Run model on test set.
        print('{} {}'.format(self.args.dataset, self.args.vec_type))

        ltmse = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        lvmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)
        print('(Last Model) Vmse {:.3f} Tmse: {:.3f}'.format(lvmse, ltmse))

        self.model.load_state_dict(self.best_mse_valid_state_dict_model)
        if self.mixer_phi is not None:
            self.mixer_phi.load_state_dict(self.best_mse_valid_state_dict_mixer_phi)

        vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)
        tmse = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Best MSE) Vmse {:.3f} Tmse: {:.3f}'.format(vmse, tmse))
        
        # for each save model -> (load model -> )
        
        # NOTE SAVE MODEL FOR TSNE
        if os.environ.get('SAVE_TSNE_MODEL', '0') == '1':
            path = f"/c2/jinakim/Drug_Discovery_j/tsne_model/tsne_model2_mNct{args.model_no_context}_RYV{os.environ.get('RANDOM_YV', '0')}_mix{args.mixer_phi}_{os.environ.get('MIX_TYPE', 'SET')}/{self.args.embed_test}/" # but embed_test should not effect result 
            os.makedirs(path, exist_ok=True)
            
            if args.mixer_phi:
                if args.seed != 42:
                    f_path = f"/c2/jinakim/Drug_Discovery_j/tsne_model/tsne_model2_mNct{args.model_no_context}_RYV{os.environ.get('RANDOM_YV', '0')}_mix{args.mixer_phi}_{os.environ.get('MIX_TYPE', 'SET')}/{self.args.embed_test}/Model_{args.sencoder}_{args.dataset}_{args.vec_type}_{args.mvalid_dataset}_{args.seed}.pth"
                else:
                    f_path = f"/c2/jinakim/Drug_Discovery_j/tsne_model/tsne_model2_mNct{args.model_no_context}_RYV{os.environ.get('RANDOM_YV', '0')}_mix{args.mixer_phi}_{os.environ.get('MIX_TYPE', 'SET')}/{self.args.embed_test}/Model_{args.sencoder}_{args.dataset}_{args.vec_type}_{args.mvalid_dataset}.pth"
                
                torch.save({
                    'model': self.model.state_dict(),
                    'mixer_phi': self.mixer_phi.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'mixer_optimizer': self.optimizermixer.state_dict() if self.optimizermixer is not None else None,
                    'args': vars(args),
                    'ltmse': ltmse,
                    'lvmse': lvmse,
                    'vmse': vmse,
                    'tmse': tmse,
                }, f_path)
            else:
                f_path = f"/c2/jinakim/Drug_Discovery_j/tsne_model/tsne_model2_mNct{args.model_no_context}_RYV{os.environ.get('RANDOM_YV', '0')}_mix{args.mixer_phi}_{os.environ.get('MIX_TYPE', 'SET')}/{self.args.embed_test}/Model_{args.sencoder}_{args.dataset}_{args.vec_type}_{args.mvalid_dataset}.pth"
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'args': vars(args),  # Save args as a dictionary,
                    'ltmse' : ltmse,
                    'lvmse' : lvmse,
                    'vmse' : vmse,
                    'tmse ': tmse, 
                }, f_path)
            
            print(f'saved >> {f_path}')
            exit()
        ######
        
        ### for plotting
        if args.tsne_plot:
            # 0. train_context
            # TODO check contextloader has remaining datasets
            raise Exception()
            set_seed(0)
            # args.batch_size = 1
            # print('args ', args)
            trainloader_test, _, mvalidloader_test, _, contextloader_test, ood1_trainloader_test, ood2_trainloader_test = get_dataset(args=args, test=True)
            
            self.test(dataloader=trainloader_test, contextloader=contextloader_test, mixer_phi=self.mixer_phi, embed_type="train_context_bunch")
            
            
            exit()
            trainloader_test, _, mvalidloader_test, _, contextloader_test, ood1_trainloader_test, ood2_trainloader_test = get_dataset(args=args, test=True)
            
            if "cX_mX" not in self.args.embed_test:
                assert self.mixer_phi is not None
                trainloader_test=InfIterator(trainloader_test)
                contextloader_test=InfIterator(contextloader_test)
                ood1_trainloader_test=InfIterator(ood1_trainloader_test)
                ood2_trainloader_test=InfIterator(ood2_trainloader_test)
                self.test(dataloader=trainloader_test, contextloader=contextloader_test, mixer_phi=self.mixer_phi, embed_type="train_context")
            # else:
            #     assert self.mixer_phi is None
            #     mvalidloader_test=InfIterator(mvalidloader_test)
            #     self.test(dataloader=mvalidloader_test, contextloader=None, mixer_phi=None, embed_type="mvalid_none")
                
            # 1. train_none
            self.test(dataloader=trainloader_test, mixer_phi=self.mixer_phi, embed_type="train_none")
            # 3. ood1_none
            self.test(dataloader=ood1_trainloader_test, mixer_phi=self.mixer_phi, embed_type="ood1_none")
            # 4. ood2_none
            self.test(dataloader=ood2_trainloader_test, mixer_phi=self.mixer_phi, embed_type="ood2_none")
        
            all_embeddings = torch.cat(self.embedding_list, dim=0)
            all_labels = np.concatenate(self.label_list, axis=0)
            all_losses = torch.cat(self.loss_list, dim=0)
            
            all_labels = torch.tensor(all_labels)
            all_losses = torch.tensor(all_losses)
                        
            all_embeddings_np = all_embeddings.numpy()
            all_labels_np = all_labels.numpy()
            all_losses_np = all_losses.numpy()
            
            path = f"/c2/jinakim/Drug_Discovery_j/analysis/tsne_last_real/{self.args.embed_test}_"
            os.makedirs(path, exist_ok=True)
            np.savez(f'/c2/jinakim/Drug_Discovery_j/analysis/tsne_last_real/{self.args.embed_test}_/{self.args.sencoder}_{self.args.dataset}_{self.args.vec_type}.npz', embeddings=all_embeddings_np, labels=all_labels_np, losses=all_losses_np)
            
            trainloader_test._iterator._shutdown_workers()
            contextloader_test._iterator._shutdown_workers()
            ood1_trainloader_test._iterator._shutdown_workers()
            ood2_trainloader_test._iterator._shutdown_workers()
        
        # save model
        # torch.save(model.state_dict(), f"{self.args.}.pth")
        
        # print('SAVED MODEL')
        # print('EXITING!!!!!!!!!!!!!')
            exit()
        return vmse, tmse, lvmse, ltmse


if __name__ == '__main__':
    args = get_arguments()

    if os.environ.get("HYPER_SWEEP", "0") == "1":
        datasets = ["hivprot", "dpp4", "nk1", ] # "hivprot", 
        featurizations = ["count", "bit"]

        arg_map = {i: (d, f) for i, (d, f) in enumerate(itertools.product(datasets, featurizations))}

        hyper_grid = {
            "lr": [1e-3,], #  1e-4
            "clr": [1e-5],
            "num_layers": [3,], #  4
            "hidden_dim": [64], # 32, 
            "optimizer": ['adamwschedulefree'],
            "n_context": [1, 4, 8],
            "dropout": [0.5],
            "inner_episodes": [10],
            "outer_episodes": [50],
            "n_mvalid": [1, 6, 16]
        }

        hyper_map = {
                i: {
                    "lr": lr,
                    "clr": clr,
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "optimizer": optimizer,
                    "n_context": n_context,
                    "dropout": dropout,
                    "inner_episodes": inner_episodes,
                    "outer_episodes": outer_episodes,
                    "n_mvalid":n_mvalid
                } for i, (lr, clr, num_layers, hidden_dim, optimizer, n_context, dropout, inner_episodes, outer_episodes, n_mvalid) \
                        in enumerate(itertools.product(*[hyper_grid[k] for k in hyper_grid.keys()]))
        }

        # print(f"{arg_map=}")
        # print(f"{hyper_map=}")
        # exit()

        path = f"experiments/hyper_search_{args.sencoder}_{os.environ.get('MIX_TYPE', 'SET')}_n_mvalid_real"
        os.makedirs(path, exist_ok=True)
        for arg_key in arg_map.keys():
            # for hyper_key in hyper_map.keys():
            dataset, featurization = arg_map[arg_key]
            args.dataset = dataset
            args.vec_type = featurization

            # set_seed(10)
            # trainloader, validloader, mvalidloader, testloader, contextloader, ood1_trainloader, ood2_trainloader = get_dataset(args=args, test=False)
            # print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
            for hyper_key in range(int(os.environ["START"]), int(os.environ["STOP"])):
                
                f_path = f"{path}/nmvalid-{dataset}-{featurization}-{hyper_key}--in{args.num_inner_dataset}.pkl"
                
                if os.path.exists(f_path):
                    # print(f"Skipping {f_path} (already exists)")
                    continue
                
                hypers = hyper_map[hyper_key]
                print(f"running with hypers: {hypers=}")
                for k, v in hypers.items():
                    setattr(args, k, v)

                set_seed(10)
                trainloader, validloader, mvalidloader, testloader, contextloader, ood1_trainloader, ood2_trainloader = get_dataset(args=args, test=False)
                print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))


                model = get_model(args=args)
                mixer_phi = get_mixer(args=args)
                
                optimizer = get_optimizer(optimizer=args.optimizer, model=model, lr=args.lr, wd=args.wd, mixer_phi=mixer_phi)
                
                optimizermixer = None

                if os.environ.get('MIX_TYPE', 'SET') not in ['MIXUP', 'MANIFOLD_MIXUP', 'SET_NO_BILEVEL', 'MIXUP_BILEVEL']: # SET, MIXUP_BILEVEL, MANIFOLD_MIXUP_BILEVEL
                    optimizermixer = None if mixer_phi is None else get_optimizer(optimizer=args.optimizer, model=mixer_phi, lr=args.clr, wd=args.cwd)


                trainer = Trainer(model=model.to(args.device), \
                                  mixer_phi=mixer_phi if mixer_phi is None else mixer_phi.to(args.device), \
                                  optimizer=optimizer, \
                                  optimizermixer=optimizermixer, \
                                  trainloader=trainloader, \
                                  validloader=validloader, \
                                  mvalidloader=mvalidloader, \
                                  contextloader=contextloader, \
                                  testloader=testloader, \
                                  ood1_trainloader=ood1_trainloader, \
                                  ood2_trainloader=ood2_trainloader, \
                                  args=args,
                                  )
                
                _, _, vmse, _ = trainer.fit()
                with open(f_path, "wb") as f:
                    pickle.dump({"mse": vmse, **hypers}, f)

        exit("exiting after hyperparameter sweep")

    losses = []
    last_losses = []

    
    # TODO need to fill in
    best_hypers = {
        ### [updated impl. stepping w self.optim, loss with L_V] ALL MIXUP BILEVEL lr=0.01, 'num_layers': 3, 'hidden_dim': 64,
         ("hivprot", "count", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 4, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 1 # , "sencoder_layer": 'max',
        },
        ("hivprot", "bit", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 16 # , "sencoder_layer": 'max',
        },
        ("dpp4", "count", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 6 # , "sencoder_layer": 'max',
        },
        ("dpp4", "bit", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 6 # , "sencoder_layer": 'max',
        },
        ("nk1", "count", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 6 # , "sencoder_layer": 'max',
        },
        ("nk1", "bit", "dsets"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'dsets', 'n_mvalid': 16 # , "sencoder_layer": 'max',
        },

        ### ALL MANIFOLD MIXUP BILEVEL lr=0.01, 'num_layers': 3, 'hidden_dim': 64,
        ("hivprot", "count", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 4, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 6 # , "sencoder_layer": 'max', 'n_mvalid': 6
        },
        ("hivprot", "bit", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 1 # , "sencoder_layer": 'max', 'n_mvalid': 1
        },
        ("dpp4", "count", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 8, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 1 # , "sencoder_layer": 'max', 'n_mvalid': 1
        },
        ("dpp4", "bit", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 8, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 1 # , "sencoder_layer": 'sum', 'n_mvalid': 1
        },
        ("nk1", "count", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 1, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 1 # "sencoder_layer": 'sum', 'n_mvalid': 1
        },
        ("nk1", "bit", "strans"): {
            'lr': 0.001, 'clr': 1e-05, 'num_layers': 3,
            'hidden_dim': 64, 'n_context': 4, 'dropout': 0.5,
            'inner_episodes': 10, 'outer_episodes': 50,
            'sencoder': 'strans', 'n_mvalid': 1 # , "sencoder_layer": 'sum', 'n_mvalid': 1
        },
    }

    hyperparams = best_hypers[(args.dataset, args.vec_type, args.sencoder)]
    for k, v in hyperparams.items():
        setattr(args, k, v)

    print(f"running {args.dataset=} {args.vec_type=} with params: {hyperparams}")

    if args.seed >= 0:
        set_seed(0)
        if os.environ.get('SAVE_TSNE_MODEL', '0') == '1' and args.seed != 42:
            args.seed = 0 # NOTE JIN using args.seed
    # args.seed = 0 # NOTE JIN using args.seed
    trainloader, validloader, mvalidloader, testloader, contextloader, ood1_trainloader, ood2_trainloader = get_dataset(args=args, test=True)

    for i in range(10):
        if i > 0:
            if args.seed >= 0:
                set_seed(i)
                if os.environ.get('SAVE_TSNE_MODEL', '0') == '1' and args.seed != 42:
                    args.seed = i # NOTE JIN using args.seed
        print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
        
        # print('args ', args)
        
        if args.tsne_plot:
            raise Exception()
        model = get_model(args=args)
        
        if args.mixer_phi:
            mixer_phi = get_mixer(args=args)
        else:
            mixer_phi = None

        optimizer = get_optimizer(optimizer=args.optimizer, model=model, lr=args.lr, wd=args.wd, mixer_phi=mixer_phi)
        
        optimizermixer = None
        
        if os.environ.get('MIX_TYPE', 'SET') not in ['MIXUP', 'MANIFOLD_MIXUP', 'SET_NO_BILEVEL', 'MIXUP_BILEVEL']: # SET, MIXUP_BILEVEL, MANIFOLD_MIXUP_BILEVEL
            optimizermixer = None if mixer_phi is None else get_optimizer(optimizer=args.optimizer, model=mixer_phi, lr=args.clr, wd=args.cwd)
        

        trainer = Trainer(model=model.to(args.device), \
                          mixer_phi=mixer_phi if mixer_phi is None else mixer_phi.to(args.device), \
                          optimizer=optimizer, \
                          optimizermixer=optimizermixer, \
                          trainloader=trainloader, \
                          validloader=validloader, \
                          mvalidloader=mvalidloader, \
                          contextloader=contextloader, \
                          testloader=testloader, \
                          ood1_trainloader=ood1_trainloader, \
                          ood2_trainloader=ood2_trainloader, \
                          args=args,
                          )

        _, tmse, _, ltmse = trainer.fit()
        losses.append(tmse)
        last_losses.append(ltmse)

    dev = os.environ["CUDA_VISIBLE_DEVICES"]
    l = np.array(losses)
    ll = np.array(last_losses)
    print(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}")
    
    os.makedirs("./experiments", exist_ok=True)
    if not args.same_setting:
        with open(f"./experiments/results-{args.sencoder}.txt", "a+") as _f:
            _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer}\n")
            _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
            _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
    else:
        if not args.mixer_phi:
            with open(f"./experiments/results-{args.sencoder}-MLP.txt", "a+") as _f:
                _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mixer_phi : {args.mixer_phi}\n")
                _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
        elif not args.exclude_mval_data_in_context:
            if args.context_dataset != None:
                with open(f"./experiments/S-results-{args.sencoder}_mvalid-all-3real-Ncd-{args.low_sim}.txt", "a+") as _f:
                    _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mv : {args.mvalid_dataset} Ncd : {args.context_dataset} sim : {args.low_sim}\n")
                    _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                    _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
            else:
                if os.environ.get('RANDOM_YV', '0') == '0':
                    with open(f"./experiments/S-results-{args.sencoder}_mvalid-all-3real-ml{args.mixing_layer}-{os.environ.get('MIXING_X_DEFAULT', 'xmix')}-mvdef{os.environ.get('MVALID_DEFAULT', '1')}-mNct{args.model_no_context}.txt", "a+") as _f:
                        _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mv : {args.mvalid_dataset} mixing_layer : {args.mixing_layer} {os.environ.get('MIXING_X_DEFAULT', 'xmix')}_test\n")
                        _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                        _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
                else:
                    if os.environ.get('MIX_TYPE', 'SET') == 'SET':
                        with open(f"./experiments/S-results-{args.sencoder}_mvalid-all-3real-ml{args.mixing_layer}-{os.environ.get('MIXING_X_DEFAULT', 'xmix')}-mvdef{os.environ.get('MVALID_DEFAULT', '1')}-mNct{args.model_no_context}-RYV{os.environ.get('RANDOM_YV', '0')}_real.txt", "a+") as _f:
                            _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mv : {args.mvalid_dataset} mixing_layer : {args.mixing_layer} {os.environ.get('MIXING_X_DEFAULT', 'xmix')}_test\n")
                            _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                            _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
                    else:
                        with open(f"./experiments/S-results-{args.sencoder}_mvalid-all-3real-ml{args.mixing_layer}-{os.environ.get('MIXING_X_DEFAULT', 'xmix')}-mvdef{os.environ.get('MVALID_DEFAULT', '1')}-mNct{args.model_no_context}-RYV{os.environ.get('RANDOM_YV', '0')}_{os.environ.get('MIX_TYPE', 'SET')}_real_3.txt", "a+") as _f:
                            _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mv : {args.mvalid_dataset} mixing_layer : {args.mixing_layer} {os.environ.get('MIXING_X_DEFAULT', 'xmix') } MIX_TYPE : {os.environ.get('MIX_TYPE', 'SET')}\n")
                            _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                            _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")
        else:
            with open(f"./experiments/S-results-{args.sencoder}_mvalid-exclude-all-3real.txt", "a+") as _f:
                _f.write(f"{args.dataset} {args.vec_type} lr: {args.lr} clr: {args.clr} {args.sencoder_layer} mv : {args.mvalid_dataset}\n")
                _f.write(f"mu: {l.mean()} +- {l.std() / np.sqrt(l.shape[0])}\n")
                _f.write(f"last performance mu: {ll.mean()} +- {ll.std() / np.sqrt(ll.shape[0])}\n\n")

    trainloader._iterator._shutdown_workers()
    validloader._iterator._shutdown_workers()
    testloader._iterator._shutdown_workers()
    
    if args.mixer_phi:
        contextloader._iterator._shutdown_workers()
    # ood1_trainloader._iterator._shutdown_workers()
    # ood2_trainloader._iterator._shutdown_workers()
