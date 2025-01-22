import os
import glob
import torch
import random
import numpy as np
import itertools
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from setenc import ContextMixer
import pickle
from arguments import get_arguments
from utils import set_seed, get_optimizer, InfIterator
from main import MLP, MLP2, initialize_weights, get_model

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



class Merck(Dataset):
    def __init__(self, split="train", vec_type='count', dataset="hivprot", is_context=False, batchsize=64):
        self.batchsize = batchsize
        self.vec_type = vec_type
        self.dataset = dataset
        self.split = split
        self.is_context = is_context
        self.data, self.labels = self.load_data()
        self.mu = 0
        self.sigma = 1

    def load_data(self):
        files = os.listdir("data/Merck/preprocessed/")

        def filter(s):
            if self.is_context:
                return self.dataset not in s.lower() and self.split in s and ".pt" in s
            return self.dataset in s.lower() and self.split in s and ".pt" in s

        files = [f for f in files if filter(f)]
        data = [torch.load(f"data/Merck/preprocessed/{f}").float() for f in files]
        max_dim = 6561

        data = [torch.cat((d, torch.zeros(d.size(0), max_dim - d.size(1))), dim=1) for d in data]
        data = torch.cat(data, dim=0)
        print(f"{data.size()=}")
        data, labels = data[:, 1:], data[:, 0]

        data = torch.exp(data) - 1

        with open("data/Merck/preprocessed/stats.pkl", "rb") as f:
            stats = pickle.load(f)

        stats = [v for v in stats if v[0].lower() == self.dataset]
        mu, sigma = stats[0][1:3]
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        print(f"{self.mu.item()=} {self.sigma.item()=}")

        # perm = torch.randperm(data.size(0))
        # return data[perm], labels[perm]
        return data, labels

    def denormalize(self, y):
        return y * self.sigma + self.mu

    def __getitem__(self, index):
        if self.is_context:
            # print("WARNING: do we need to permute things here?")
            data = self.data[index * self.batchsize: (index + 1) * self.batchsize]
            return data

        if self.vec_type == "count":
            return self.data[index], self.labels[index]
        elif self.vec_type == "bit":
            return (self.data[index] > 0).float(), self.labels[index]
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.is_context:
            return len(self.data) // self.batchsize
        return len(self.data)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(args):
    trainset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False)
    validset = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=False)

    train_idx = np.load(f"data/perms/train-idx-{args.dataset}.npy")
    val_idx = np.load(f"data/perms/val-idx-{args.dataset}.npy")

    trainset.data = trainset.data[train_idx]
    trainset.labels = trainset.labels[train_idx]
    validset.data = validset.data[val_idx]
    validset.labels = validset.labels[val_idx]

    testset = Merck(split="test", vec_type=args.vec_type, dataset=args.dataset, is_context=False)

    m = trainset.data.amax()
    trainset.data = trainset.data / m
    validset.data = validset.data / m
    testset.data = testset.data / m

    g = torch.Generator()
    g.manual_seed(0)
    shuffle=False

    trainloader = DataLoader(
        trainset,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
            # generator=g,
            shuffle=shuffle,
            pin_memory=True
    )
    validloader = DataLoader(
        validset,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
            # generator=g,
            shuffle=False,
            pin_memory=True
    )

    testloader = DataLoader(testset, \
                            batch_size=args.batch_size, \
                            num_workers=args.num_workers, \
                            worker_init_fn=seed_worker, \
                            # generator=g,
                            shuffle=False, pin_memory=True)

    contextloader = None
    if args.mixer_phi:
        contextloader = Merck(split="train", vec_type=args.vec_type, dataset=args.dataset, is_context=True, batchsize=args.batch_size * args.n_context)
        contextloader.data = contextloader.data / m
    return trainloader, validloader, testloader, contextloader


class Trainer:
    def __init__(
        self, epochs=500,
        model=None,
        mixer_phi=None,
        optimizer=None,
        testloader=None,
        trainloader=None,
        validloader=None,
        contextloader=None,
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
        self.mixer_phi = mixer_phi
        self.contextloader = contextloader
        self.optimizermixer = optimizermixer

    def train(self):
        losses = []
        self.model.train()
        self.optimizer.train()
        for x, y in self.trainloader:
            y_hat = self.model(x=x.to(self.args.device))
            loss = F.mse_loss(y_hat.squeeze(), y.to(self.args.device))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)

    def calc_loss(self, y_hat, y, test=False):
        if len(y_hat.size()) > 1:
            raise ValueError(f"sizes should not happen: {y_hat.size()=} {y.size()=}")
        return F.mse_loss(y.cuda().squeeze(), y_hat.cuda().squeeze())

        # y_hat_real = y_hat[:, 0]
        # loss = F.mse_loss(y, y_hat_real)

        # y_hat_unlabeled = y_hat[:, 1:].reshape(-1)
        # mu = y_hat_unlabeled.mean()
        # logvar = y_hat_unlabeled.var().log()

        # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # loss += 0.01 * kl
        # return loss

    def test(self, dataloader, mixer_phi=None):
        self.model.eval()
        self.optimizer.eval()
        if mixer_phi is not None:
            mixer_phi.eval()

        with torch.no_grad():
            losses = []
            counts = 0
            for x, y in dataloader:
                y_hat = self.model(x=x.to(self.args.device), mixer_phi=mixer_phi)

                y = y.cuda().squeeze()
                y_hat = y_hat.cuda().squeeze()
                # y = self.trainloader.dataset.dataset.denormalize(y)
                # y_hat = self.trainloader.dataset.dataset.denormalize(y_hat)

                loss = self.calc_loss(y_hat, y, test=True)
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

            v3 = torch.autograd.grad(d_LT_dw, lmbda(), grad_outputs=v2, retain_graph=True)
            d_LV_dlmbda = torch.autograd.grad(L_V, lmbda())
            return [d - v for d, v in zip(d_LV_dlmbda, v3)]

        def train_mixer(model, optimizer, mixer_phi, x, y, context, device, interp_loss=False):
            model.train()
            mixer_phi.train()
            if optimizer is not None:
                optimizer.train()

            x, y, context = x.to(device), y.to(device), context.to(device)

            # 1. Mix context with labeled sample x
            y_hat_mixed = model(x=x, context=context, mixer_phi=mixer_phi)
            loss = self.calc_loss(y_hat_mixed.squeeze(), y.squeeze())

            # 2. Pass unmixed sample through model
            # y_hat = model(x=x, context=None, mixer_phi=mixer_phi)
            # loss = loss + F.cross_entropy(y_hat, y, weight=self.trainloader.dataset.classweights.to(self.args.device))

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
        # validloader = InfIterator(self.validloader)

        self.optimizermixer.train()
        for episode in tqdm(range(self.args.outer_episodes), ncols=75, leave=False):
            tlosses = []
            for k in tqdm(range(self.args.inner_episodes), ncols=75, leave=False):
                x, y = next(trainloader)
                context = self.contextloader[torch.randperm(len(self.contextloader))[0]]

                train_loss = train_mixer(model=self.model, optimizer=self.optimizer, mixer_phi=self.mixer_phi, \
                                         x=x, y=y, context=context, device=self.args.device, interp_loss=True)
                tlosses.append(train_loss.item())

            # Compute hypergradients
            x_t, y_t = next(trainloader)  # self.trainloader.dataset.get_batch(batch_size=50*self.args.train_batch_size)
            context_t = self.contextloader[torch.randperm(len(self.contextloader))[0]]
            L_T = train_mixer(model=self.model, optimizer=None, mixer_phi=self.mixer_phi, x=x_t, y=y_t, \
                              context=context_t, device=self.args.device)

            # self.model.eval()
            # self.mixer_phi.eval()
            # x_v, y_v = next(validloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)
            x_v, y_v = next(trainloader)  # self.validloader.dataset.get_batch(batch_size=self.args.BS*self.args.batch_size)

            context_v = None  # self.contextloader[torch.randperm(len(self.contextloader))[0]].to(self.args.device)
            y_v_hat = self.model(x=x_v.to(self.args.device), context=context_v, mixer_phi=self.mixer_phi)

            L_V = self.calc_loss(y_v_hat.squeeze(), y_v.to(self.args.device).squeeze(), test=False)  # , weight=self.validloader.dataset.classweights.to(self.args.device))

            hgrads = hypergradients(L_V=L_V, L_T=L_T, lmbda=self.mixer_phi.parameters, w=self.model.parameters, i=5, alpha=self.args.lr)

            self.optimizermixer.zero_grad()
            for p, g in zip(self.mixer_phi.parameters(), hgrads):
                hypergrad = torch.clamp(g, -5.0, 5.0)
                hypergrad *= 1.0 - (episode / (self.args.outer_episodes))
                p.grad = hypergrad
            self.optimizermixer.step()

            # Run model on validation set.
            vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)

            print('Episode: {:<3} tloss: {:.3f} vmse: {:.3f}'.format(\
                episode, np.mean(tlosses), vmse))

            if vmse < best_vmse:
                best_vmse = vmse
                episodes_without_improvement = 0
                self.best_mse_valid_state_dict_model = deepcopy(self.model.state_dict())
                self.best_mse_valid_state_dict_mixer_phi = deepcopy(self.mixer_phi.state_dict())
            else:
                episodes_without_improvement += 1
                if episodes_without_improvement == self.args.early_stopping_episodes:
                    break

    def fit(self):
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
            self.train_mixer_phi()

        # Run model on test set.
        print('{} {}'.format(self.args.dataset, self.args.vec_type))

        tmse = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)
        print('(Last Model) Vmse {:.3f} Tmse: {:.3f}'.format(vmse, tmse))

        self.model.load_state_dict(self.best_mse_valid_state_dict_model)
        if self.mixer_phi is not None:
            self.mixer_phi.load_state_dict(self.best_mse_valid_state_dict_mixer_phi)

        vmse = self.test(dataloader=self.validloader, mixer_phi=self.mixer_phi)
        tmse = self.test(dataloader=self.testloader, mixer_phi=self.mixer_phi)
        print('(Best MSE) Vmse {:.3f} Tmse: {:.3f}'.format(vmse, tmse))
        return vmse, tmse


if __name__ == '__main__':
    args = get_arguments()

    do_hyper_search = False
    if do_hyper_search:
        datasets = ["hivprot", "dpp4", "nk1"]
        featurizations = ["count", "bit"]

        arg_map = {i: (d, f) for i, (d, f) in enumerate(itertools.product(datasets, featurizations))}

        hyper_grid = {
            "lr": [1e-3, 1e-4],
            "clr": [1e-3, 1e-4],
            "num_layers": [2, 4],
            "hidden_dim": [32, 64],
            "ctx_points": [1, 8, 16, 32],
            "dropout": [0.2, 0.5],
            "inner_episodes": [20, 50],
            "outer_episodes": [30, 50],
        }

        hyper_map = {
                i: {
                    "lr": lr,
                    "clr": clr,
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "ctx_points": ctx_points,
                    "dropout": dropout,
                    "inner_episodes": inner_episodes,
                    "outer_episodes": outer_episodes,
                } for i, (lr, clr, num_layers, hidden_dim, ctx_points, dropout, inner_episodes, outer_episodes) \
                        in enumerate(itertools.product(*[hyper_grid[k] for k in hyper_grid.keys()]))
        }

        print(f"{arg_map=}")
        print(f"{hyper_map=}")
        exit()

        path = "experiments/hyper_search"
        os.makedirs(path, exist_ok=True)
        for arg_key in arg_map.keys():
            # for hyper_key in hyper_map.keys():
            for hyper_key in range(int(os.environ["START"]), int(os.environ["STOP"])):
                dataset, featurization = arg_map[arg_key]
                args.dataset = dataset
                args.vec_type = featurization

                hypers = hyper_map[hyper_key]
                for k, v in hypers.items():
                    setattr(args, k, v)

                set_seed(10)
                trainloader, validloader, testloader, contextloader = get_dataset(args=args)
                print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
                model, mixer_phi = get_model(args=args)

                optimizer = get_optimizer(optimizer=args.optimizer, model=model, lr=args.lr, wd=args.wd)
                optimizermixer = None if mixer_phi is None else get_optimizer(optimizer=args.optimizer, model=mixer_phi, lr=args.clr, wd=args.cwd)

                trainer = Trainer(model=model.to(args.device), \
                                  mixer_phi=mixer_phi if mixer_phi is None else mixer_phi.to(args.device), \
                                  optimizer=optimizer, \
                                  optimizermixer=optimizermixer, \
                                  trainloader=trainloader, \
                                  validloader=validloader, \
                                  contextloader=contextloader, \
                                  testloader=testloader, \
                                  args=args,
                                  )

                vmse, _ = trainer.fit()
                with open(f"{path}/{dataset}-{featurization}-{hyper_key}.pkl", "wb") as f:
                    pickle.dump({"mse": vmse, **hypers}, f)

    for i in range(10):
        set_seed(i)
        trainloader, validloader, testloader, contextloader = get_dataset(args=args)
        print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
        model, mixer_phi = get_model(args=args)

        optimizer = get_optimizer(optimizer=args.optimizer, model=model, lr=args.lr, wd=args.wd)
        optimizermixer = None if mixer_phi is None else get_optimizer(optimizer=args.optimizer, model=mixer_phi, lr=args.clr, wd=args.cwd)

        trainer = Trainer(model=model.to(args.device), \
                          mixer_phi=mixer_phi if mixer_phi is None else mixer_phi.to(args.device), \
                          optimizer=optimizer, \
                          optimizermixer=optimizermixer, \
                          trainloader=trainloader, \
                          validloader=validloader, \
                          contextloader=contextloader, \
                          testloader=testloader, \
                          args=args,
                          )

        _, tmse = trainer.fit()
        exit("do something with the final tmse here, save it in a file with the mean and std error")
