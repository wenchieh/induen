import numpy as np
from numpy.random import MT19937, RandomState
from scipy.sparse.linalg import norm
from sklearn.utils.extmath import squared_norm
from sklearn.preprocessing import normalize

""" @ is equal to np.matmul()
    * is equal to np.multiply()
""" 

# Calculate the reconstruction error as equation (9)
def reconstrcutionError(As:list, Cs:dict, beta, facMat, lambdaMat, sigmaMat, GG):
    error = 0
    layer = len(As)
    for i in range(layer):
        error += squared_norm(As[i] - facMat[i] @ lambdaMat[i] @ facMat[i].T)
        for j in range(layer):
            if j > i and GG[i, j] != 0:
                error += beta[i, j] * squared_norm(Cs[(i, j)] - facMat[i] @ sigmaMat[(i, j)] @ facMat[j].T)
    return error

""" Coupled non-negative matrix factorization used in Corduen
    As : list of within-layer adjacency matrix 
    Cs : dict of cross-layer dependency matrix
    GG : the structure of the dependency in multi-layered network
    facMat: list of latent factor matrix for each layer
    lambdaMat: list of diagnoal matrix 
    sigmaMat: dict of diagnoal matrix
    beta: the dispersion parameter representing the coupling strength
"""
def CNMF(As: list, Cs: dict, GG: np.ndarray, beta=None, R=10, epoch=50, reg=1e-10, seed=12345):
    rs = RandomState(seed)
    mt19937 = MT19937()
    mt19937.state = rs.get_state()
    layer = len(As)
    facMat = []         # factor matrices
    lambdaMat = []      # Lambda matrices
    sigmaMat = {}       # Sigma  matrices

    # Set beta as equation (8) heuristicly. (i.e. beta==None)
    if type(beta)!=np.ndarray: 
        beta = np.zeros((layer, layer))
        for i in range(layer):
            for j in range(layer):
                if GG[i, j] != 0 and j > i:
                    # Calculate beta_{ij} as equation (8)
                    beta[i, j] = (norm(As[i])**2 + norm(As[j])**2) / (2*norm(Cs[(i, j)])**2)
                    beta[j, i] = beta[i, j]


    # Initialize the factor matrix with column_norm = 1
    # Initialize the lambda and sigma matrix as identity matrix
    
    for i in range(layer):
        facMat.append(normalize(rs.rand(As[i].shape[0], R), axis=0))
        lambdaMat.append(np.identity(R))
        for j in range(layer):
            if GG[i, j] != 0 and j > i:
                sigmaMat[(i, j)] = np.identity(R)
                sigmaMat[(j, i)] = sigmaMat[(i, j)]

    # totalErr = [reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG)]

    for t in range(epoch):
        # Update factor matrix Ui as equation (10)
        for i in range(layer):
            upper = 2 * As[i] @ facMat[i] @ lambdaMat[i]
            lower_tmp = 2 * lambdaMat[i] @ (facMat[i].T @ facMat[i]) @ lambdaMat[i]
            for j in range(layer):
                if GG[i, j] != 0:
                    upper += beta[i, j] * Cs[(i, j)] @ facMat[j] @ sigmaMat[(i, j)]
                    lower_tmp += beta[i, j] * sigmaMat[(i, j)] @ (facMat[j].T @ facMat[j]) @ sigmaMat[(i, j)]
            lower = facMat[i] @ lower_tmp + reg * np.ones(facMat[i].shape)
            multiplier = np.power(np.divide(upper, lower), 1/2)
            facMat[i] = np.multiply(facMat[i], multiplier)
            ui = facMat[i].copy()
            facMat[i], norm_ui = normalize(facMat[i], axis=0, return_norm=True)
            
            # Update lambda matrix as equation (7)
            lambda_upper = ui.T @ As[i] @ ui
            reuse_item = ui.T @ ui
            lambda_lower = reuse_item @ lambdaMat[i] @ reuse_item + reg*np.ones((R, R))
            lambda_multiplier = np.power(np.divide(lambda_upper, lambda_lower), 1/2)
            lambdaMat[i] = np.multiply(lambdaMat[i], lambda_multiplier)
            lambdaMat[i] = np.multiply(lambdaMat[i], np.multiply(norm_ui, norm_ui))

            # Update sigma matrix as equation (7)
            for j in range(layer):
                if GG[i, j] != 0:
                    sigma_upper = ui.T @ Cs[(i, j)] @ facMat[j]
                    sigma_lower = reuse_item @ sigmaMat[(i, j)] @ (facMat[j].T @ facMat[j]) + reg*np.ones((R, R))
                    sigma_multiplier = np.power(np.divide(sigma_upper, sigma_lower), 1/2)
                    sigmaMat[(i, j)] = np.multiply(sigmaMat[(i, j)], sigma_multiplier)
                    sigmaMat[(i, j)] = np.multiply(sigmaMat[(i, j)], norm_ui)
                    sigmaMat[(j, i)] = sigmaMat[(i, j)].T

        # totalErr.append(reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG))
    return facMat, lambdaMat, sigmaMat, beta


""" Coupled Orthogonal Non-negative Matrix Factorization used in Corduen
    Extend the ONMF proposed by Chris Ding to coupled setting.
    As : list of within-layer adjacency matrix 
    Cs : dict of cross-layer dependency matrix
    GG : the structure of the dependency in multi-layered network
    facMat: list of latent factor matrix for each layer
    lambdaMat: list of diagnoal matrix 
    sigmaMat: dict of diagnoal matrix
    beta: the dispersion parameter representing the coupling strength
"""


def CONMF(As: list, Cs: dict, GG:np.ndarray, beta=None,R=10,epoch=50,reg=1e-10,seed=12345):
    rs = RandomState(seed)
    mt19937 = MT19937()
    mt19937.state = rs.get_state()
    layer = len(As)
    facMat = []         # factor matrices
    lambdaMat = []      # Lambda matrices
    sigmaMat = {}       # Sigma  matrices

    # Set beta as equation (8) heuristicly. (i.e. beta==None)
    if type(beta)!=np.ndarray: 
        beta = np.zeros((layer, layer))
        for i in range(layer):
            for j in range(layer):
                if GG[i, j] != 0 and j > i:
                    # Calculate beta_{ij} as equation (8)
                    beta[i, j] = (norm(As[i])**2 + norm(As[j])**2) / (2*norm(Cs[(i, j)])**2)
                    beta[j, i] = beta[i, j]

    # Initialize the factor matrix with column_norm = 1
    # Initialize the lambda and sigma matrix as identity matrix

    for i in range(layer):
        facMat.append(normalize(rs.rand(As[i].shape[0], R), axis=0))
        lambdaMat.append(np.identity(R))
        for j in range(layer):
            if GG[i, j] != 0 and j > i:
                sigmaMat[(i, j)] = np.identity(R)
                sigmaMat[(j, i)] = sigmaMat[(i, j)]

    # totalErr = [reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG)]

    for t in range(epoch):
        # Update factor matrix Ui as equation (10)
        for i in range(layer):
            upper = 2 * As[i] @ facMat[i] @ lambdaMat[i]
            for j in range(layer):
                if GG[i, j] != 0:
                    upper += beta[i, j] * Cs[(i, j)] @ facMat[j] @ sigmaMat[(i, j)]
            lower = facMat[i] @ (facMat[i].T@upper) + reg * np.ones(facMat[i].shape)
            multiplier = np.power(np.divide(upper, lower), 1/2)
            facMat[i] = np.multiply(facMat[i], multiplier)
            ui = facMat[i].copy()
            facMat[i], norm_ui = normalize(facMat[i], axis=0, return_norm=True)

            # Update lambda matrix as equation (7)
            lambda_upper = ui.T @ As[i] @ ui
            reuse_item = ui.T @ ui
            lambda_lower = reuse_item @ lambdaMat[i] @ reuse_item + reg*np.ones((R, R))
            lambda_multiplier = np.power(np.divide(lambda_upper, lambda_lower), 1/2)
            lambdaMat[i] = np.multiply(lambdaMat[i], lambda_multiplier)
            lambdaMat[i] = np.multiply(lambdaMat[i], np.multiply(norm_ui, norm_ui))

            # Update sigma matrix as equation (7)
            for j in range(layer):
                if GG[i, j] != 0:
                    sigma_upper = ui.T @ Cs[(i, j)] @ facMat[j]
                    sigma_lower = reuse_item @ sigmaMat[(i, j)] @ (facMat[j].T @ facMat[j]) + reg*np.ones((R, R))
                    sigma_multiplier = np.power(np.divide(sigma_upper, sigma_lower), 1/2)
                    sigmaMat[(i, j)] = np.multiply(sigmaMat[(i, j)], sigma_multiplier)
                    sigmaMat[(i, j)] = np.multiply(sigmaMat[(i, j)], norm_ui)
                    sigmaMat[(j, i)] = sigmaMat[(i, j)].T

        # totalErr.append(reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG))
    return facMat, lambdaMat, sigmaMat, beta
