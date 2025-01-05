import os
import glob
import torch
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

__all__ = ['get_dataset']

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
        #data = []
        #for filepath in filepaths:
        #    data.append(torch.load(filepath))
        #data = torch.cat(data, dim=0)
        #print(data.size()); exit()
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
                np.asarray(compute_class_weight(class_weight="balanced", classes=np.unique(self.labels.numpy()), y=self.labels.numpy())))

    def load_dataset(self):
        datapath = os.path.join(self.root, \
                'antimalarial_data_processed',\
                split_types[self.split_type],\
                fingerprints[self.fingerprint],\
                '2', '{}.pth'.format(self.split)
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

def get_dataset(args):
    if args.dataset == 'antimalaria':
        trainset = AntiMalaria(root=args.root, split='train', \
                split_type=args.split_type, \
                fingerprint=args.fingerprint)
        validset = AntiMalaria(root=args.root, split='valid', \
                split_type=args.split_type, \
                fingerprint=args.fingerprint)
        testset = AntiMalaria(root=args.root, split='test', \
                split_type=args.split_type, \
                fingerprint=args.fingerprint)
    
    else:
        raise NotImplementedError
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    #g = torch.Generator()
    #g.manual_seed(0)

    trainloader = DataLoader(trainset, \
            batch_size=args.batch_size, \
            num_workers=args.num_workers, \
            #worker_init_fn=seed_worker, \
            #generator=g,
            shuffle=True, pin_memory=True)
    validloader = DataLoader(validset, \
            batch_size=args.batch_size, \
            num_workers=args.num_workers, \
            #worker_init_fn=seed_worker, \
            #generator=g,
            shuffle=False, pin_memory=True)
    testloader = DataLoader(testset, \
            batch_size=args.batch_size, \
            num_workers=args.num_workers, \
            #worker_init_fn=seed_worker, \
            #generator=g,
            shuffle=False, pin_memory=True)
    
    contextloader = None
    if args.mixer_phi:
        contextloader = ZINC(batchsize=args.batch_size, fingerprint=args.fingerprint)
        #contextloader = DataLoader(contextdata, \
        #        batch_size=args.batch_size, \
        #        num_workers=args.num_workers, \
        #        shuffle=True, pin_memory=True)
    return trainloader, validloader, testloader, contextloader

if __name__ == '__main__':
    args = lambda: None
    args.root = 'data'
    args.num_workers = 4
    args.batch_size = 128
    args.fingerprint = 'rdkit'
    args.dataset = 'antimalaria'
    args.split_type = 'spectral'
    args.mixer_phi = None
    
    trainloader, validloader, testloader, _ = get_dataset(args=args)
    labels = []
    for x, y in trainloader:
        print(x.size())
    #y = torch.cat(labels, dim=0)
