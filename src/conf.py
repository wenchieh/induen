import numpy as np
from numpy.random import MT19937, RandomState
from scipy.sparse.linalg import norm
from sklearn.utils.extmath import squared_norm
from sklearn.preprocessing import normalize

""" 
    @ is equal to np.matmul()
    * is equal to np.multiply()
""" 

# Calculate the reconstruction error as equation (9)
def reconstr_error(As:list,Cs:dict,GG:np.ndarray,factors,lambdas,sigmas,beta):
    loss = 0
    layer = len(As)
    for i in range(layer):
        loss += squared_norm(As[i] - factors[i] @ lambdas[i] @ factors[i].T)
        for j in range(i+1, layer):
            if GG[i, j] != 0:
                loss += beta[i, j] * squared_norm(Cs[(i, j)] - factors[i] @ sigmas[(i, j)] @ factors[j].T)
    return loss

# Calculate the objective proposed in the paper.
def objective(As:list,Cs:list,GG:np.ndarray,factors,lambdas,sigmas,beta,reg):
    loss = 0
    layer = len(As)
    rank = factors[0].shape[0]
    for i in range(layer):
        loss += squared_norm(As[i] - factors[i] @ lambdas[i] @ factors[i].T)
        loss += reg * np.sum(np.abs(factors[i]))
        loss += reg * sum([abs(lambdas[i][k,k]) for k in rank])
        
        for j in range(i+1, layer):
            if GG[i, j] != 0:
                loss += beta[i, j] * squared_norm(Cs[(i, j)] - factors[i] @ sigmas[(i, j)] @ factors[j].T)
                loss += reg * sum([abs(sigmas[j][k,k]) for k in rank])
    return loss

# Calculate the betas.
def get_betas(As:list,Cs:dict,GG:np.ndarray):
    layer = len(As)
    beta = np.zeros((layer, layer))
    for i in range(layer):
        for j in range(i+1, layer):
            if GG[i, j] != 0:
                # Calculate beta_{i,j} as equation (8)
                beta[i, j] = (norm(As[i])**2 + norm(As[j])**2) / (2*norm(Cs[(i, j)])**2)
                beta[j, i] = beta[i, j]
    return beta


""" 
    Coupled non-negative matrix factorization used in Corduen
    As : list of within-layer adjacency matrix 
    Cs : dict of cross-layer dependency matrix
    GG : the structure of the dependency in multi-layered network
    facMat: list of latent factor matrix for each layer
    lambdaMat: list of diagnoal matrix 
    sigmaMat: dict of diagnoal matrix
    beta: the dispersion parameter representing the coupling strength
"""

def coupled_nmf(As: list, Cs: dict, GG: np.ndarray, beta=None, R=10, epoch=50, reg=1e-10, seed=1234):
    rs = RandomState(seed)
    mt19937 = MT19937()
    mt19937.state = rs.get_state()
    layer = len(As)
    facMat = []         # factor matrices
    lambdaMat = []      # Lambda matrices
    sigmaMat = {}       # Sigma  matrices

    # Set beta as equation (8) heuristicly. (i.e. beta==None)
    if not isinstance(beta, np.ndarray):
        beta = get_betas(As,Cs,GG)

    # Initialize the factor matrix with column_norm = 1
    # Initialize the lambda and sigma matrix as identity matrix
    
    factors = [normalize(rs.rand(As[i].shape[0], R), axis=0) for i in range(layer)]
    lambdas = [np.identity(R) for _ in range(layer)]
    sigmas = {k:np.identity(R) for k in Cs.keys()}

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
            facMat[i] = facMat[i] * multiplier
            ui = facMat[i].copy()
            facMat[i], norm_ui = normalize(facMat[i], axis=0, return_norm=True)
            
            # Update lambda matrix as equation (7)
            lambda_upper = ui.T @ As[i] @ ui
            reuse_item = ui.T @ ui
            lambda_lower = reuse_item @ lambdaMat[i] @ reuse_item + reg*np.ones((R, R))
            lambda_multiplier = np.power(np.divide(lambda_upper, lambda_lower), 1/2)
            lambdaMat[i] = lambdaMat[i] * lambda_multiplier
            lambdaMat[i] = lambdaMat[i] * (norm_ui * norm_ui)

            # Update sigma matrix as equation (7)
            for j in range(layer):
                if GG[i, j] != 0:
                    sigma_upper = ui.T @ Cs[(i, j)] @ facMat[j]
                    sigma_lower = reuse_item @ sigmaMat[(i, j)] @ (facMat[j].T @ facMat[j]) + reg*np.ones((R, R))
                    sigma_multiplier = np.power(np.divide(sigma_upper, sigma_lower), 1/2)
                    sigmaMat[(i,j)] = sigmaMat[(i,j)] * sigma_multiplier
                    sigmaMat[(i,j)] = sigmaMat[(i,j)] * norm_ui
                    sigmaMat[(j, i)] = sigmaMat[(i, j)].T

        # totalErr.append(reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG))
    return facMat, lambdaMat, sigmaMat, beta


""" 
    Coupled Orthogonal Non-negative Matrix Factorization used in Corduen
    Extend the ONMF proposed by Chris Ding to coupled setting.
    As : list of within-layer adjacency matrix 
    Cs : dict of cross-layer dependency matrix
    GG : the structure of the dependency in multi-layered network
    facMat: list of latent factor matrix for each layer
    lambdaMat: list of diagnoal matrix 
    sigmaMat: dict of diagnoal matrix
    beta: the dispersion parameter representing the coupling strength
"""



def coupled_orthogonal_nmf(As: list, Cs: dict, GG:np.ndarray, beta=None, R=10, epoch=50, reg=1e-10, seed=12345):
    rs = RandomState(seed)
    mt19937 = MT19937()
    mt19937.state = rs.get_state()
    layer = len(As)
    factors = []      # factor matrices
    lambdas = []      # lambda matrices
    sigmas = {}       # sigma  matrices

    # Set beta as equation (8) heuristicly. (i.e. beta==None)
    if not isinstance(beta, np.ndarray):
        beta = get_betas(As,Cs,GG)

    # Initialize the factor matrix with column_norm = 1
    # Initialize the lambda and sigma matrix as identity matrix

    factors = [normalize(rs.rand(As[i].shape[0], R), axis=0) for i in range(layer)]
    lambdas = [np.identity(R) for _ in range(layer)]
    sigmas = {k:np.identity(R) for k in Cs.keys()}

    """
    for i in range(layer):
        factors.append(normalize(rs.rand(As[i].shape[0], R), axis=0))
        lambdas.append(np.identity(R))
        for j in range(i+1, layer):
            if GG[i, j] != 0:
                sigmas[(i, j)] = np.identity(R)
                sigmas[(j, i)] = sigmas[(i, j)]
    """

    # totalErr = [reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG)]

    for t in range(epoch):
        # Update factor matrix U_{i} as equation (10)
        for i in range(layer):
            upper = 2 * As[i] @ factors[i] @ lambdas[i]
            for j in range(layer):
                if GG[i, j] != 0:
                    upper += beta[i, j] * Cs[(i, j)] @ factors[j] @ sigmas[(i, j)]
            lower = factors[i] @ (factors[i].T@upper) + reg * np.ones(factors[i].shape)
            multiplier = np.power(np.divide(upper, lower), 1/2)
            factors[i] = factors[i] * multiplier
            ui = factors[i].copy()
            factors[i], norm_ui = normalize(factors[i], axis=0, return_norm=True)

            # Update lambda matrix as equation (7)
            lambda_upper = ui.T @ As[i] @ ui
            reuse_item = ui.T @ ui
            lambda_lower = reuse_item @ lambdas[i] @ reuse_item + reg*np.ones((R, R))
            lambda_multiplier = np.power(np.divide(lambda_upper, lambda_lower), 1/2)
            lambdas[i] = lambdas[i] * lambda_multiplier

            # Absorb the column norm of Ui into Lambda
            lambdas[i] = lambdas[i] * (norm_ui * norm_ui)

            # Update sigma matrix as equation (7)
            for j in range(layer):
                if GG[i, j] != 0:
                    sigma_upper = ui.T @ Cs[(i, j)] @ factors[j]
                    sigma_lower = reuse_item @ sigmas[(i, j)] @ (factors[j].T @ factors[j]) + reg*np.ones((R, R))
                    sigma_multiplier = np.power(np.divide(sigma_upper, sigma_lower), 1/2)
                    sigmas[(i, j)] = sigmas[(i, j)] * sigma_multiplier

                    # Absorb the column norm of Ui into Sigma
                    sigmas[(i, j)] = sigmas[(i, j)] * norm_ui
                    sigmas[(j, i)] = sigmas[(i, j)].T

        # totalErr.append(reconstrcutionError(As, Cs, beta, facMat, lambdaMat, sigmaMat, GG))
    return factors, lambdas, sigmas, beta
