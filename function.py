import time, os, random
import torch
import numpy as np
import math
from scipy.stats import ortho_group, special_ortho_group
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
from scipy import interp
from scipy.ndimage import filters
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from torchvision import datasets

# ==================================================== BL-EM  ==================================================== #
pi = torch.tensor(math.pi)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def GMM_logprobs_Full(x, pi_q, mu_q, Sigma_q, mask=None):
    '''
    K number of components, N number of samples, D dimension of each observation
    param x:    N,D
    param pi_q: K,1 or N,K
    param mu_q: K,D
    param std_q: K,D,D
    param mask: N,D
    return:
    '''
    N, D = x.shape

    t1 = torch.log(pi_q).squeeze()  # K or N,K
    t2 = - D / 2. * torch.log(2. * pi)  #
    t3 = - 0.5 * torch.logdet(Sigma_q)  # K
    x_mu = x.unsqueeze(1) - mu_q.unsqueeze(0)  # N,K,D
    t4 = - 0.5 * torch.matmul(
        x_mu.unsqueeze(-2),
        torch.matmul(torch.inverse(Sigma_q), x_mu.unsqueeze(-1))
    ).squeeze(-1).squeeze(-1)  # N,K

    log_Normal = t2 + t3 + t4  # [N, K]
    log_piNormal = t1 + log_Normal  # [N, K]
    log_gmm = torch.logsumexp(log_piNormal, dim=1, keepdim=True)  # N,1

    return log_gmm, log_piNormal, log_Normal


def big_GMM_logprobs_Full(x, pi_q, mu_q, Sigma_q, chunk_size, mask=None):
    '''
    K number of components, N number of samples, D dimension of each observation
    :param x:    N,D
    :param pi_q: K,1 or N,K
    :param mu_q: K,D
    :param std_q: K,D,D
    :param mask: N,D
    :return:
    '''
    N, D = x.shape

    t1 = torch.log(pi_q).squeeze()  # K or N,K
    t2 = - D / 2. * torch.log(2. * pi)  #
    t3 = - 0.5 * torch.logdet(Sigma_q)  # K
    x_mu = x.unsqueeze(1) - mu_q.unsqueeze(0)  # N,K,D

    invSigma_q = torch.inverse(Sigma_q)
    x_mu_chunks = torch.chunk(x_mu, int(N / chunk_size), dim=0)
    t4 = torch.cat([
        - 0.5 * torch.matmul(
            x_mu1.unsqueeze(-2),
            torch.matmul(invSigma_q, x_mu1.unsqueeze(-1))
        ).squeeze(-1).squeeze(-1)
        for x_mu1 in x_mu_chunks
    ], dim=0)
    
    log_Normal = t2 + t3 + t4  # [N, K]
    log_piNormal = t1 + log_Normal  # [N, K]
    log_gmm = torch.logsumexp(log_piNormal, dim=1, keepdim=True)  # N,1

    del t1, t2, t3, t4, x_mu, x_mu_chunks

    return log_gmm, log_piNormal, log_Normal


def rand_mat(dim=2, method='orthogonal'):
    if method == 'orthogonal':
        A = torch.tensor(ortho_group.rvs(dim), dtype=torch.float32)  # Orthogonal
    elif method == 'rotation':
        A = torch.tensor(special_ortho_group.rvs(dim), dtype=torch.float32)  # Rotation
    else:
        print('Invalid input for <method>')
    return A

def correct_Sigma(Sigma_p, eps):

    if isinstance(Sigma_p, torch.Tensor):
        L, V = torch.linalg.eigh(Sigma_p)
        Sigma_p = V @ torch.diag_embed(
            torch.maximum(torch.tensor(eps), L)
        ) @ torch.linalg.inv(V)
        
    elif isinstance(Sigma_p, np.ndarray):
        L, V = np.linalg.eigh(Sigma_p)
        Sigma_p = V @ np.diag(
            np.maximum(eps, L)
        ) @ np.linalg.inv(V)
    else:
        raise ValueError("Sigma_p should be a tensor or numpy array")
    
    return Sigma_p


def load_data(dataset_name, train_path, test_path, data_type, split_data, device, random_seed):
    # Load data
    # first column: class 
    # normalize train_data using the min-max map to the interval [0, 1]
    seed_torch(random_seed)
    
    scaler = MinMaxScaler()

    if data_type == 'mat':

        train_data_dict  = scipy.io.loadmat(train_path)
        train_data = pd.DataFrame(train_data_dict['DataSet'])
        raw_X_train = train_data.iloc[:,1:]  # feature
        raw_y_train = train_data.iloc[:,0]  # class

        if dataset_name == 'Vehicle':
            raw_X_train = train_data.iloc[:,1:].astype(np.uint8)  # feature
            raw_y_train = train_data.iloc[:,0].astype(np.uint8)  # class

        X_train = scaler.fit_transform(raw_X_train)  # array
        y_train = raw_y_train.values

        if test_path is not None:
            test_data_dict  = scipy.io.loadmat(test_path)
            test_data = pd.DataFrame(test_data_dict['DataSet'])
            raw_X_test = test_data.iloc[:,1:]  # feature
            raw_y_test = test_data.iloc[:,0]  # class

            X_test = scaler.fit_transform(raw_X_test)  # array
            y_test = raw_y_test.values

        elif split_data:
            # split test data
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed)
        else:
            X_test = X_train
            y_test = y_train

 
    elif data_type == 'svmlib':

        raw_X_train, raw_y_train = load_svmlight_file(train_path)
        mat = raw_X_train.todense() 
        raw_X_train = pd.DataFrame(mat)
        raw_y_train = pd.DataFrame(raw_y_train)

        X_train = scaler.fit_transform(raw_X_train)  # array
        y_train = raw_y_train.values.flatten() - 1

        if test_path is not None:
            raw_X_test, raw_y_test = load_svmlight_file(test_path)
            mat = raw_X_test.todense() 
            raw_X_test = pd.DataFrame(mat)
            raw_y_test = pd.DataFrame(raw_y_test)

            X_test = scaler.fit_transform(raw_X_test)  # array
            y_test = raw_y_test.values.flatten() - 1

        elif split_data:
            # split test data
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed)
        else:
            X_test = X_train
            y_test = y_train
        
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device) 
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)  
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return X_train, X_test, y_train, y_test, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

