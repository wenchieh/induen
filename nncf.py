import numpy as np
from numpy.random import MT19937, RandomState
# from scipy.sparse.linalg import norm
from util import norm
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.utils.extmath import squared_norm
from sklearn.preprocessing import normalize

# Non-negative Coupled matrix factorization
# As : list of csc_matrix
# Cs : each value is a csc_matrix
# GG : a matrix to represent the dependency structure


def NNCF(As: list, Cs: dict, GG, epoch=50, R=10, reg=1e-6, seed=12345):
    rs = RandomState(seed)
    mt19937 = MT19937()
    mt19937.state = rs.get_state()
    layer = len(As)
    facmat = []         # factor matrices
    LambdaMat = []      # Lambda matrices
    SigmaMat = {}       # Sigma  matrices
    alpha = np.zeros((layer, layer))
    for i in range(layer):
        # initialize the latent matrix u_i with column_norm = 1
        facmat.append(normalize(rs.rand(As[i].shape[0], R), axis=0))
        LambdaMat.append(np.identity(R))
        # calculate the parameter alpha_ij for each dependency matrix
        for j in range(layer):
            if GG[i, j] != 0 and j > i:
                # the dispersion parameter representing the coupling strength
                alpha[i, j] = (norm(As[i])**2+norm(As[j])**2) / \
                    (2*norm(Cs[(i, j)])**2)
                alpha[j, i] = alpha[i, j]
                SigmaMat[(i, j)] = np.identity(R)
                SigmaMat[(j, i)] = SigmaMat[(i, j)]
    # totalErr = [reconstrcutionError(As,Cs,alpha,facmat,Lambda,Sigma,GG)]
    for t in range(epoch):
        for i in range(layer):
            upper = 2 * As[i] @ facmat[i] @ LambdaMat[i]
            lower_tmp = 2 * \
                LambdaMat[i] @ (facmat[i].T @ facmat[i]) @ LambdaMat[i]
            # lower = 2 * facmat[i] @ LambdaMat[i] @ (facmat[i].T @ facmat[i]) @ LambdaMat[i] + reg * np.ones(facmat[i].shape)
            for j in range(layer):
                if GG[i, j] != 0:
                    upper += alpha[i, j] * \
                        Cs[(i, j)] @ facmat[j] @ SigmaMat[(i, j)]
                    lower_tmp += alpha[i, j] * SigmaMat[(i, j)] @ (
                        facmat[j].T @ facmat[j]) @ SigmaMat[(i, j)]
                    # lower += alpha[i, j] * facmat[i] @ SigmaMat[(i,j)] @ (facmat[j].T @ facmat[j]) @ SigmaMat[(i,j)]
            lower = facmat[i] @ lower_tmp + reg * np.ones(facmat[i].shape)
            multiplier = np.power(np.divide(upper, lower), 1/2)
            # update factor matrix U_i by entrywise product
            facmat[i] = np.multiply(facmat[i], multiplier)
            ui = facmat[i].copy()
            facmat[i], norm_ui = normalize(facmat[i], axis=0, return_norm=True)
            # norm吸收进入lambda、sigma矩阵
            lambda_upper = ui.T @ As[i] @ ui
            reuse_item = ui.T @ ui
            lambda_lower = reuse_item @ LambdaMat[i] @ reuse_item + \
                reg*np.ones((R, R))
            lambda_multiplier = np.power(
                np.divide(lambda_upper, lambda_lower), 1/2)
            LambdaMat[i] = np.multiply(LambdaMat[i], lambda_multiplier)
            LambdaMat[i] = np.multiply(
                LambdaMat[i], np.multiply(norm_ui, norm_ui))
            for j in range(layer):
                if GG[i, j] != 0:
                    sigma_upper = ui.T @ Cs[(i, j)] @ facmat[j]
                    sigma_lower = reuse_item @ SigmaMat[(i, j)] @ (
                        facmat[j].T @ facmat[j]) + reg*np.ones((R, R))
                    sigma_multiplier = np.power(
                        np.divide(sigma_upper, sigma_lower), 1/2)
                    SigmaMat[(i, j)] = np.multiply(
                        SigmaMat[(i, j)], sigma_multiplier)
                    SigmaMat[(i, j)] = np.multiply(SigmaMat[(i, j)], norm_ui)
                    SigmaMat[(j, i)] = SigmaMat[(i, j)].T
        # totalErr.append(reconstrcutionError(As,Cs,alpha,facmat,Lambda,Sigma,GG))
    return facmat, LambdaMat, SigmaMat, alpha
