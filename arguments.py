import torch
import argparse
from utils import str2bool

__all__ = ['get_arguments']


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=0, type=int, help='number of times to repeat experiment.')
    parser.add_argument('--seed', default=42, type=int, help='number of times to repeat experiment.')
    parser.add_argument('--root', default="/c2/jinakim/dataset_backup/antimalaria/", type=str, help='modelzoo to use for training.')
    parser.add_argument('--model', default='mlp', type=str, help='model to fit nn to [mlp].')
    parser.add_argument('--dataset', default='antimalaria', type=str, help='dataset used to train zoo.')
    parser.add_argument('--split_type', default='spectral', type=str, help='dataset split type.')
    parser.add_argument('--fingerprint', default='ecfp', type=str, help='dataset figerprint type.')
    parser.add_argument('--batch_size', default=128, type=int, help='batchsize.')
    parser.add_argument('--num_outputs', default=2, type=int, help='number of properties to predict.')
    parser.add_argument('--sencoder', default='strans', type=str, help='set encoder to use')
    parser.add_argument('--sencoder_layer', default='pma', type=str, help='layer type in the set encoder')

    parser.add_argument('--vec_type', default='count', type=str, help='bit or count vector (for Merck)')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs.')
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
    parser.add_argument('--ln', default=False, type=str2bool, help='batchnorm.')
    parser.add_argument('--initialize_weights', default=False, type=str2bool, help='init weigths or not.')

    # Parameters for hyperparameter optimization
    parser.add_argument('--mixer_phi', default=False, type=str2bool, help='use context mixer or not.')
    parser.add_argument('--outer_episodes', type=int, default=100, help='outer episodes for BO')
    parser.add_argument('--inner_episodes', type=int, default=5, help='inner episodes for BO')
    parser.add_argument('--n_context', type=int, default=16, help='number of context points')
    parser.add_argument('--early_stopping_episodes', type=int, default=10, help='inner episodes for BO')
    parser.add_argument('--num_inner_dataset', type=int, default=1, help='num_inner_dataset')
    parser.add_argument('--same_setting', action='store_true')
    parser.add_argument('--low_sim', type=str)
    parser.add_argument('--exclude_mval_data_in_context', action='store_true')
    
    parser.add_argument('--mvalid_dataset', nargs='+', type=str)
    parser.add_argument('--context_dataset', nargs='+', type=str)
    parser.add_argument('--specify_ood_dataset', nargs='+', type=str)
    parser.add_argument('--embed_test',  default="ours_best", type=str, help='embed type.')
    parser.add_argument('--tsne_plot', action='store_true')
    parser.add_argument('--mixing_layer', default=0, type=int, help='number of properties to predict.')
    
    parser.add_argument('--model_no_context', action='store_true')
    # parser.add_argument('--mixup_epochs', default=300, type=int, help='number of training epochs.')
    parser.add_argument('--n_mvalid', default=-1, type=int, help='n_mvalid')
    
    

    

    args = parser.parse_args()

    if args.fingerprint == 'rdkit':
        args.in_features = 2042
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args
