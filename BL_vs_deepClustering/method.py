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
import numpy as np
import scipy
from scipy import interp
from scipy.ndimage import filters
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import gc
from function import *

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

bl_epsilon = torch.tensor(1e-5)


# ==================================================== BL-EM  ==================================================== #

def bl_EM(num_component, x_dim, X_train_tensor, y_train_tensor, X_train, y_train, Niter, NITnei, eps, alpha_pi, a_beta, b_beta, 
          Amode, Amethod, device, data_size, chunk_size, random_seed, txt_dir, kmeans_init, P1, P2):

    seed_torch(random_seed)

    BigLearnITS , BigLearnTheta = [], []
    BigLearnNMI_testS = []
    BigLearnAcc_testS = []

    # initialize
    if kmeans_init:
        n_samples, _ = X_train.shape
        resp = np.zeros((n_samples, num_component))
        label = (
                cluster.KMeans(
                    n_clusters=num_component, n_init=1, random_state=random_seed
                )
                .fit(X_train)
                .labels_
            )
        resp[np.arange(n_samples), label] = 1

        qzgx = torch.tensor(resp, dtype=X_train_tensor.dtype, device=X_train_tensor.device)

        mask_s, mask_t = torch.zeros(x_dim, dtype=torch.int), torch.ones(x_dim, dtype=torch.int)
        Tindx = mask_t > 0
        Tx = torch.where(Tindx)[0]  # K,D,D
        Tx1, Tx2, Tx3 = torch.meshgrid(torch.arange(num_component), Tx, Tx)
        
        pi_p = qzgx.mean(0).unsqueeze(-1)  # K,1
        mu_p = (qzgx.unsqueeze(2) * X_train_tensor[:, Tindx].unsqueeze(1)).mean(0) / torch.maximum(bl_epsilon, pi_p)  # K,D
        x_mu = X_train_tensor[:, Tindx].unsqueeze(1) - mu_p[:, Tindx].unsqueeze(0)  # N,K,D
        
        N = qzgx.shape[0]
        x_mu_chunks = torch.chunk(x_mu, int(N / chunk_size), dim=0)
        qzgx_chunks = torch.chunk(qzgx, int(N / chunk_size), dim=0)
        tmp = 0
        for x_mu1, qzgx1 in zip(x_mu_chunks, qzgx_chunks):
            tmp += (qzgx1.unsqueeze(2).unsqueeze(3) * torch.matmul(x_mu1.unsqueeze(-1), x_mu1.unsqueeze(-2))
                    ).sum(0)  # K,D,D
        Sigma_p = tmp / N / torch.maximum(bl_epsilon, pi_p).unsqueeze(-1)  # K,D,D
        Sigma_p = correct_Sigma(Sigma_p, eps)

    else:
        pi_p = 1. / num_component * torch.ones(num_component, 1, device=device) # π: Initialized to a uniform distribution
        mu_p = torch.randn(num_component, x_dim, device=device) # μ: random initialization
        Sigma_p = torch.diag_embed(torch.ones(num_component, x_dim, device=device)) # ∑: 1 (Diagonal Matrix)SS

    accmax = 0
    nmimax = 0
    for IT in range(Niter):

        urand = torch.rand(1)
        # urand = 0.8
        if urand < P1: # joint 0.4
            case = 0
            mask_s, mask_t = torch.zeros(x_dim, dtype=torch.int), torch.ones(x_dim, dtype=torch.int)
        elif urand < P2: # Marginal 0.5
            case = 1
            mask_s, mask_t = torch.zeros(x_dim, dtype=torch.int), torch.zeros(x_dim, dtype=torch.int)
            num_ones = int(x_dim * Beta(a_beta, b_beta).sample())
            indices = torch.randperm(x_dim)[:num_ones]
            mask_t[indices] = 1    
        else:   # Marginal + Rotation
            case = 2

            if Amode == 'rand':
                A = rand_mat(dim=x_dim, method=Amethod)  # rotation orthogonal
                A = A.to(device)

            mask_s, mask_t = torch.zeros(x_dim, dtype=torch.int), torch.zeros(x_dim, dtype=torch.int)
            num_ones = int(x_dim * Beta(a_beta, b_beta).sample())
            indices = torch.randperm(x_dim)[:num_ones]
            mask_t[indices] = 1 

        if case == 0 or case == 1:  # Joint, Marginal
            Tindx = mask_t > 0
            Tx = torch.where(Tindx)[0]  # K,D,D
            Tx1, Tx2, Tx3 = torch.meshgrid(torch.arange(mu_p.shape[0]), Tx, Tx)

            for iii in range(NITnei):

                if data_size == 'small':
                    # E
                    log_gmm, log_piNormal, _ = GMM_logprobs_Full(X_train_tensor[:, Tindx], pi_p, mu_p[:, Tindx],
                                                                    Sigma_p[Tx1, Tx2, Tx3])  # N,1  N,K
                    qzgx = (log_piNormal - log_gmm).exp()  # N,K

                    # M
                    # update π: [K,1]
                    pi_p1 = qzgx.mean(0).unsqueeze(-1)
                    # update μ: [K,D]    
                    mu_p[:, Tindx] = (qzgx.unsqueeze(2) * X_train_tensor[:, Tindx].unsqueeze(1)
                                        ).mean(0) / torch.maximum(bl_epsilon, pi_p1)
                    # update ∑: [K,D,D]  
                    x_mu = X_train_tensor[:, Tindx].unsqueeze(1) - mu_p[:, Tindx].unsqueeze(0)  # N,K,D
                    Sigma_p[Tx1, Tx2, Tx3] = (qzgx.unsqueeze(2).unsqueeze(3) *
                                                torch.matmul(x_mu.unsqueeze(-1), x_mu.unsqueeze(-2))
                                                ).mean(0) / torch.maximum(bl_epsilon, pi_p1).unsqueeze(-1)  # K,D,D

                    Sigma_p = correct_Sigma(Sigma_p, eps)

                    #  the MAP estimate of π
                    pi_p = (pi_p1 + alpha_pi)
                    pi_p = pi_p / pi_p.sum()

                elif data_size == 'big':
                    # E
                    log_gmm, log_piNormal, _ = big_GMM_logprobs_Full(X_train_tensor[:, Tindx], pi_p, mu_p[:, Tindx], Sigma_p[Tx1, Tx2, Tx3], chunk_size=chunk_size)  # N,1  N,K
                    qzgx = (log_piNormal - log_gmm).exp()  # N,K
                    # M
                    pi_p1 = qzgx.mean(0).unsqueeze(-1)  # K,1
                    mu_p[:, Tindx] = (qzgx.unsqueeze(2) * X_train_tensor[:, Tindx].unsqueeze(1)).mean(0) / torch.maximum(bl_epsilon, pi_p1)  # K,D
                    x_mu = X_train_tensor[:, Tindx].unsqueeze(1) - mu_p[:, Tindx].unsqueeze(0)  # N,K,D
                    
                    N = qzgx.shape[0]
                    x_mu_chunks = torch.chunk(x_mu, int(N / chunk_size), dim=0)
                    qzgx_chunks = torch.chunk(qzgx, int(N / chunk_size), dim=0)
                    tmp = 0
                    for x_mu1, qzgx1 in zip(x_mu_chunks, qzgx_chunks):
                        tmp += (qzgx1.unsqueeze(2).unsqueeze(3) * torch.matmul(x_mu1.unsqueeze(-1), x_mu1.unsqueeze(-2))
                                ).sum(0)  # K,D,D
                    Sigma_p[Tx1, Tx2, Tx3] = tmp / N / torch.maximum(bl_epsilon, pi_p1).unsqueeze(-1)  # K,D,D

                    Sigma_p = correct_Sigma(Sigma_p, eps)

                    #  the MAP estimate of π
                    pi_p = (pi_p1 + alpha_pi)
                    pi_p = pi_p / pi_p.sum()


        elif case == 2 :  # Marginal + Rotation/Orthogonal
            Tindx = mask_t > 0
            Tx = torch.where(Tindx)[0]  # K,D,D
            Tx1, Tx2, Tx3 = torch.meshgrid(torch.arange(mu_p.shape[0]), Tx, Tx)

            # transformation
            y_all = X_train_tensor.mm(A.t())
            bar_mu_p = mu_p.mm(A.t())
            bar_Sigma_p = torch.matmul(A.unsqueeze(0), torch.matmul(Sigma_p, A.t()))

            for iii in range(NITnei):

                if data_size == 'small':
                    # E - A space
                    log_gmm, log_piNormal, _ = GMM_logprobs_Full(y_all[:, Tindx], pi_p, bar_mu_p[:, Tindx],
                                                                    bar_Sigma_p[Tx1, Tx2, Tx3])  # N,1  N,K
                    qzgy = (log_piNormal - log_gmm).exp()  # N,K

                    # M - A space
                    # update π: [K,1]
                    pi_p1 = qzgy.mean(0).unsqueeze(-1)  
                    # update μ: [K,D]
                    bar_mu_p[:, Tindx] = (qzgy.unsqueeze(2) * y_all[:, Tindx].unsqueeze(1)
                                            ).mean(0) / torch.maximum(bl_epsilon, pi_p1)  # Can assign values
                    # update ∑: [K,D,D]
                    y_mu = y_all[:, Tindx].unsqueeze(1) - bar_mu_p[:, Tindx].unsqueeze(0)  # N,K,D

                    bar_Sigma_p[Tx1, Tx2, Tx3] = (qzgy.unsqueeze(2).unsqueeze(3) *
                                                    torch.matmul(y_mu.unsqueeze(-1), y_mu.unsqueeze(-2))
                                                    ).mean(0) / torch.maximum(bl_epsilon, pi_p1).unsqueeze(-1)  # K,D,D

                    
                    bar_Sigma_p = correct_Sigma(bar_Sigma_p, eps)

                    # transfer to original space
                    mu_p = bar_mu_p.mm(A)
                    Sigma_p = torch.matmul(A.t().unsqueeze(0), torch.matmul(bar_Sigma_p, A))

                    Sigma_p = correct_Sigma(Sigma_p, eps)

                    #  the MAP estimate of π
                    pi_p = (pi_p1 + alpha_pi)
                    pi_p = pi_p / pi_p.sum()

                elif data_size == 'big':
                    # E - A space
                    log_gmm, log_piNormal, _ = big_GMM_logprobs_Full(y_all[:, Tindx], pi_p, bar_mu_p[:, Tindx],
                                                                    bar_Sigma_p[Tx1, Tx2, Tx3], chunk_size=chunk_size)  # N,1  N,K
                    qzgy = (log_piNormal - log_gmm).exp()  # N,K

                    # M - A space
                    # update π: [K,1]
                    pi_p1 = qzgy.mean(0).unsqueeze(-1)  
                    # update μ: [K,D]
                    bar_mu_p[:, Tindx] = (qzgy.unsqueeze(2) * y_all[:, Tindx].unsqueeze(1)
                                            ).mean(0) / torch.maximum(bl_epsilon, pi_p1)  # Can assign values
                    # update ∑: [K,D,D]
                    y_mu = y_all[:, Tindx].unsqueeze(1) - bar_mu_p[:, Tindx].unsqueeze(0)  # N,K,D

                    N = qzgy.shape[0]
                    y_mu_chunks = torch.chunk(y_mu, int(N / chunk_size), dim=0)
                    qzgy_chunks = torch.chunk(qzgy, int(N / chunk_size), dim=0)
                    tmp = 0
                    for y_mu1, qzgy1 in zip(y_mu_chunks, qzgy_chunks):
                        tmp += (qzgy1.unsqueeze(2).unsqueeze(3) * torch.matmul(y_mu1.unsqueeze(-1), y_mu1.unsqueeze(-2))
                                ).sum(0)  # K,D,D
                    bar_Sigma_p[Tx1, Tx2, Tx3] = tmp / N / torch.maximum(bl_epsilon, pi_p1).unsqueeze(-1)  # K,D,D

                    bar_Sigma_p = correct_Sigma(bar_Sigma_p,eps)           

                    # transfer to original space
                    mu_p = bar_mu_p.mm(A)
                    Sigma_p = torch.matmul(A.t().unsqueeze(0), torch.matmul(bar_Sigma_p, A))

                    Sigma_p = correct_Sigma(Sigma_p,eps)

                    #  the MAP estimate of π
                    pi_p = (pi_p1 + alpha_pi)
                    pi_p = pi_p / pi_p.sum()


        # predict (E step)
        if data_size == 'small':

            test_log_gmm, test_log_piNormal, _ = GMM_logprobs_Full(X_train_tensor, pi_p, mu_p,
                                                Sigma_p)  # N,1  N,K
            test_qzgx = (test_log_piNormal - test_log_gmm).exp()  # N,K
            test_predict_labels = test_qzgx.argmax(axis=1)

        elif data_size == 'big':
            if X_train_tensor.shape[0] <= chunk_size:
                test_log_gmm, test_log_piNormal, _ = big_GMM_logprobs_Full(X_train_tensor, pi_p, mu_p,
                                                        Sigma_p, chunk_size=X_train_tensor.shape[0])  # N,1  N,K
            else:
                test_log_gmm, test_log_piNormal, _ = big_GMM_logprobs_Full(X_train_tensor, pi_p, mu_p,
                                                    Sigma_p, chunk_size=chunk_size)  # N,1  N,K
            test_qzgx = (test_log_piNormal - test_log_gmm).exp()  # N,K
            test_predict_labels = test_qzgx.argmax(axis=1)

        # ACC & NMI
        acc = cluster_acc(y_train_tensor, test_predict_labels.cpu().numpy())
        BigLearnAcc_testS.append(acc)
        nmi = nmi_score(y_train_tensor, test_predict_labels.cpu().numpy())
        BigLearnNMI_testS.append(nmi)

        if acc > accmax:
            accmax = acc
        if nmi > nmimax:
            nmimax = nmi

        BigLearnTheta.append((pi_p, mu_p, Sigma_p))

        with open(txt_dir, "a") as f:
            f.write(f'seed={random_seed}, IT={IT}, Current-NMI={nmi}, Current-ACC={acc}, Max-NMI={nmimax}, Max-ACC={accmax}\n')

        BigLearnITS.append(IT)
        
    return BigLearnITS, BigLearnTheta, BigLearnNMI_testS, BigLearnAcc_testS, nmimax, accmax