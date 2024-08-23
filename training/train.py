import torch
import random
import argparse
import numpy as np

from utils import str2bool
from training import Trainer
from models import get_model
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--run', default=0, type=int, help='number of times to repeat experiment.')
parser.add_argument('--root', default='data', type=str, help='modelzoo to use for training.')
parser.add_argument('--model', default='mlp', type=str, help='model to fit nn to [mlp].')
parser.add_argument('--dataset', default='antimalaria', type=str, help='dataset used to train zoo.')
parser.add_argument('--split_type', default='spectral', type=str, help='dataset split type.')
parser.add_argument('--fingerprint', default='ecfp', type=str, help='dataset figerprint type.')
parser.add_argument('--batch_size', default=128, type=int, help='batchsize.')
parser.add_argument('--num_outputs', default=2, type=int, help='number of properties to predict.')

parser.add_argument('--epochs', default=50, type=int, help='number of training epochs.')
parser.add_argument('--optimizer', default='adamw', type=str, help='dataset used to train zoo.')
parser.add_argument('--num_workers', default=8, type=int, help='trainloader number of workers.')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate.')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay.')

parser.add_argument('--hidden_dim', default=32, type=int, help='dimension to project each weight element.')
parser.add_argument('--in_features', default=2048, type=int, help='dimension to input features.')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers.')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate.')
parser.add_argument('--batchnorm', default=True, type=str2bool, help='batchnorm.')

#Parameters for hyperparameter optimization
parser.add_argument('--contextmixer', default=False, type=str2bool, help='use context mixer or not.')
parser.add_argument('--outer_episodes', type=int, default=100, help='outer episodes for BO')
parser.add_argument('--inner_episodes', type=int, default=5, help='inner episodes for BO')
parser.add_argument('--early_stopping_episodes', type=int, default=10, help='inner episodes for BO')

args = parser.parse_args()

if __name__ == '__main__':
    #seed = 332
    #np.random.seed(seed)
    #random.seed(seed)
    #torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainloader, validloader, testloader, contextloader = get_dataset(args=args)
    print('Trainset: {} ValidSet: {} TestSet: {}'.format(len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
    model, contextmixer = get_model(args=args)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizermixer = None if contextmixer is None else torch.optim.AdamW(contextmixer.parameters(), lr=args.lr, weight_decay=args.wd)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizermixer = None if contextmixer is None else torch.optim.Adam(contextmixer.parameters(), lr=1e-4) #args.lr)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #optimizermixer = None if contextmixer is None else torch.optim.SGD(contextmixer.parameters(), lr=1e-4) #args.lr)

    trainer = Trainer(model=model.to(args.device), \
            contextmixer=contextmixer if contextmixer is None else contextmixer.to(args.device), \
            optimizer=optimizer, \
            optimizermixer = optimizermixer, \
            trainloader=trainloader, \
            validloader=validloader, \
            contextloader=contextloader, \
            testloader=testloader, \
            args=args
            )

    trainer.fit()
