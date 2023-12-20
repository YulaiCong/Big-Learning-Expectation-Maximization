import gzip
import time, os, random
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.distributions.categorical import Categorical
import matplotlib as mpl
from scipy.stats import ortho_group, special_ortho_group
import copy
from sklearn import metrics, cluster
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io
from torch.distributions.beta import Beta
from sklearn.cluster import KMeans
import matplotlib.animation as manimation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
from scipy import interp
from scipy.ndimage import filters
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from zmq import device
# from six.moves import cPickle
from torch.utils.data import Dataset


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
    t4 = - 0.5 * torch.matmul(
        x_mu.unsqueeze(-2),
        torch.matmul(torch.inverse(Sigma_q), x_mu.unsqueeze(-1))
    ).squeeze(-1).squeeze(-1)  # N,K

    log_Normal = t2 + t3 + t4  # [N, K]
    log_piNormal = t1 + log_Normal  # [N, K]
    log_gmm = torch.logsumexp(log_piNormal, dim=1, keepdim=True)  # N,1

    if torch.isnan(log_gmm).any():
        print("GMM_logprobs_Full: NaN detected")
    else:
        print("GMM_logprobs_Full: No NaN")

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

    # Optimize : part_1 = torch.matmul(torch.inverse(Sigma_q), x_mu.unsqueeze(-1))   # N,K,D,1
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

    if torch.isnan(log_gmm).any():
        print("Warning: NaN detected")

    del t1, t2, t3, t4, x_mu, x_mu_chunks

    return log_gmm, log_piNormal, log_Normal


def rand_mat(dim=2, method='orthogonal'):
    # method = 'rotation', 'orthogonal'
    if method == 'orthogonal':
        A = torch.tensor(ortho_group.rvs(dim), dtype=torch.float32)  # Orthogonal
    elif method == 'rotation':
        A = torch.tensor(special_ortho_group.rvs(dim), dtype=torch.float32)  # Rotation
    else:
        print('Invalid input for <method>')
    return A

def correct_Sigma(Sigma_p, eps): 

    if isinstance(Sigma_p, torch.Tensor):
        Sigma_p = Sigma_p + 1e-10 * torch.eye(Sigma_p.size(1), device=Sigma_p.device).unsqueeze(0)
        L, V = torch.linalg.eigh(Sigma_p)
        Sigma_p = V @ torch.diag_embed(
            torch.maximum(torch.tensor(eps), L)
        ) @ torch.linalg.inv(V)
    
    return Sigma_p


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment 

    # ind = linear_assignment(w.max() - w)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) / y_pred.size


def load_fashion():
    
    train_dataset = datasets.FashionMNIST(root='./dataset', train=True, download=False)
    test_dataset = datasets.FashionMNIST(root='./dataset', train=False, download=False)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X = np.concatenate((X_train,X_test))
    Y = np.concatenate((y_train,y_test))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print(('Fashion samples', X.shape))
    return X, Y
    
def LoadDatasetByName(dataset_name):
    if dataset_name == 'Fashion':
        x, y = load_fashion()
    return x, y

class LoadDataset(Dataset):

    def __init__(self, dataset_name):
        self.x, self.y = LoadDatasetByName(dataset_name)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))