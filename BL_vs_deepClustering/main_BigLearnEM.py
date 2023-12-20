from __future__ import print_function, division
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
from torch.utils.data import Dataset
from function import load_fashion, cluster_acc
import warnings
import time
from pathlib import Path
warnings.filterwarnings("ignore")
from method import *


class LoadDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_fashion()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

 
def train_BL(args, random_seed):   
    
    # Cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(args.device)

    alpha_pi = 1. /  args.num_component
    Amethod = 'orthogonal'  # 'orthogonal', 'rotation'
    Amode = 'rand'  # rand  fix10 

    BigLearnITS, BigLearnTheta, BigLearnNMI_testS, BigLearnAcc_testS, nmimax, accmax = bl_EM( args.num_component,
                                                                                            args.x_dim, 
                                                                                            data, # [:40000,:]
                                                                                            y, # [:40000]
                                                                                            dataset.x, # array[:40000]
                                                                                            y, # [:40000]
                                                                                            args.Niter, 
                                                                                            args.NITnei, 
                                                                                            args.eps, 
                                                                                            alpha_pi, 
                                                                                            args.a_beta, 
                                                                                            args.b_beta, 
                                                                                            Amode, 
                                                                                            Amethod, 
                                                                                            args.device,
                                                                                            args.data_size,
                                                                                            args.chunk_size,
                                                                                            random_seed,
                                                                                            args.txt_dir,
                                                                                            args.kmeans_init,
                                                                                            args.P1,
                                                                                            args.P2
                                                                                            )
    
    torch.save((random_seed, BigLearnTheta, args.num_component, args.NITnei, BigLearnITS,BigLearnNMI_testS, BigLearnAcc_testS, nmimax, accmax),
                        args.out_dir + f'eps_{args.eps}_N_{args.Niter}_nei_{args.NITnei}_{args.kmeans_init}_P1{args.P1}P2{args.P2}_{random_seed}.pt')
    
    return accmax, nmimax

def sci_notation_to_float(s):
    return float(s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='BigLearn-EM training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='Fashion', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_component', default=10, type=int)
    parser.add_argument('--x_dim', default=784, type=int)
    parser.add_argument('--Niter', default=200, type=int)
    parser.add_argument('--NITnei', default=5, type=int)
    parser.add_argument('--eps', default=1e-2, type=sci_notation_to_float)
    parser.add_argument('--a_beta', default=5, type=int)
    parser.add_argument('--b_beta', default=1, type=int)        
    parser.add_argument('--data_size', default='big', type=str)
    parser.add_argument('--chunk_size', default=50, type=int) 
    parser.add_argument('--out_dir', default='./results', type=str, help='different datasets have different out_dir')
    parser.add_argument('--txt_dir', default='./results', type=str, help='different datasets have different out_dir')
    parser.add_argument('--kmeans_init', action='store_true')
    parser.add_argument('--P1', default=0.4, type=float)
    parser.add_argument('--P2', default=0.5, type=float)


    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available()
    # print("use cuda: {}".format(args.cuda))
    # device = torch.device("cuda" if args.cuda else "cpu")
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.txt_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == 'Fashion':
        args.pretrain_path = ''
        args.n_clusters = 10
        args.n_z = 50
        args.n_input = 784
        args.num_sample = 70000
        dataset = LoadDataset() 
    print(args)
    bestacc = 0 
    bestnmi = 0

    random_seedS = [30]
    for i in range(len(random_seedS)):

        args.txt_dir = args.txt_dir + f'{random_seedS[i]}_eps_{args.eps}_N_{args.Niter}_nei_{args.NITnei}_{args.kmeans_init}_P1{args.P1}P2{args.P2}' + '.txt'

        acc, nmi = train_BL(args, random_seedS[i])
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))
