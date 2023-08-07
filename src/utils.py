import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csc_matrix

# Check a matrix symmetric or not.
def symmetric(x, tol=1e-20):
    if x.shape[0] != x.shape[1]:
        return False
    else:
        resi = (x.T - x).tocsc() if sp.issparse(x) else x.T - x
        if resi.min()>-1*tol and resi.max()<tol:
            return True
    return False

# Check a graph has loop or not.
def hasloop(graph:lil_matrix):
    for i in range(graph.shape[0]):
        if graph[i,i]:
            return True
    return False

# Norm for both dense and sparse matrix.
def norm(x, ord=None):
    if sp.issparse(x):
        return sp.linalg.norm(x,ord)
    else:
        return np.linalg.norm(x,ord)
    
# Generate a random matrix.
def random_matrix(size, prob):
    a = np.random.random(size)
    return np.where(a<prob, 1, 0)

# Convert the multilayered network matrix to lil_matrix
def convert_to_lil(As: list, Cs: dict):
    newAs = [lil_matrix(adj) for adj in As]
    newCs = {k:lil_matrix(cs) for k,cs in Cs.items()}
    return newAs, newCs

# Convert the multilayered network matrix to csc_matrix
def convert_to_csc(As: list, Cs: dict):
    newAs = [csc_matrix(adj) for adj in As]
    newCs = {k:csc_matrix(cs) for k,cs in Cs.items()}
    return newAs, newCs

# Sample a multilayered subgraph via candidate nodes
def sample(As: list, Cs: dict, candidate: list):
    for i in range(len(candidate)):
        if len(candidate[i]) == 0:
            raise ValueError('The candidate list is empty!')
        if max(candidate[i]) >= As[i].shape[0]:
            raise ValueError('The candidate list is out of range!')
        
    layer = len(As)
    newAs = [As[i][candidate[i], :][:, candidate[i]] for i in range(layer)]
    newCs = {(i,j):Cs[(i,j)][candidate[i], :][:, candidate[j]] for (i,j) in Cs.keys()}
    return newAs, newCs

# Aggregate the given multilayer network and weights into a block matrix.
def aggregate(As:list, Cs:dict, GG:np.ndarray, gammas=None):
    if not isinstance(gammas, np.ndarray):
        gammas = np.ones_like(GG, dtype=np.float64)
    layer = len(As)
    index = [0] + [adj.shape[0] for adj in As]
    pos = np.cumsum(index)
    cur_size = sum(index)
    block = lil_matrix((cur_size, cur_size))

    for i in range(layer): 
        for a,b in zip(As[i].nonzero()[0], As[i].nonzero()[1]):
            x, y = pos[i]+a, pos[i]+b # coordinates of block matrix.
            block[x, y] = gammas[i,i] * As[i][a, b]

        for j in range(i+1, layer):
            if GG[i, j] != 0:
                for a,b in zip(Cs[(i, j)].nonzero()[0], Cs[(i, j)].nonzero()[1]):
                    x, y = pos[i]+a, pos[j]+b
                    block[x, y] = gammas[i,j] * Cs[(i, j)][a, b]
                    block[y, x] = block[x,y]
    return block, pos

# Indicator of whether there exist empty res.
def indicator(nodes:list):
    flag = 1
    for i in range(len(nodes)):
        flag *= int(len(nodes[i])>0)
    return flag

# Select candidate nodes via the factor.
def select_candidate(factor, nsize, col):
    n = factor.shape[0]  # number of nodes
    thre = np.sqrt(1/n)  # threshold
    topk  = int(0.02* n*(1 - n/nsize)) # empirical top-k
    cand_thre = sorted(np.argwhere(factor[:, col] > thre)[:, 0])
    cand_topk = sorted(np.argpartition(factor[:,col], -1*topk)[-1*topk:])
    candidate = cand_thre if len(cand_thre)>len(cand_topk) else cand_topk
    return candidate

