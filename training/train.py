import torch
import random
import argparse
import numpy as np

from models import get_model
from datasets import get_dataset
from training import Trainer
from utils import str2bool, set_seed, get_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--run', default=0, type=int, help='number of times to repeat experiment.')
parser.add_argument('--seed', default=42, type=int, help='number of times to repeat experiment.')
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
parser.add_argument('--clr', default=1e-3, type=float, help='mixer_phi learning rate.')
parser.add_argument('--cwd', default=1e-5, type=float, help='mixer_phi weight decay.')

parser.add_argument('--hidden_dim', default=32, type=int, help='dimension to project each weight element.')
parser.add_argument('--in_features', default=2048, type=int, help='dimension to input features.')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers.')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate.')
parser.add_argument('--batchnorm', default=True, type=str2bool, help='batchnorm.')
parser.add_argument('--initialize_weights', default=False, type=str2bool, help='init weigths or not.')

#Parameters for hyperparameter optimization
parser.add_argument('--mixer_phi', default=False, type=str2bool, help='use context mixer or not.')
parser.add_argument('--outer_episodes', type=int, default=100, help='outer episodes for BO')
parser.add_argument('--inner_episodes', type=int, default=5, help='inner episodes for BO')
parser.add_argument('--early_stopping_episodes', type=int, default=20, help='inner episodes for BO')

args = parser.parse_args()

if __name__ == '__main__':
    #set_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
