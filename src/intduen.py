import numpy as np
from scipy.sparse import lil_matrix,csc_matrix

from src.baselines import greedyCharikar
from src.boosting import fastExpander
from src.nncf import CONF 


# Return the block matrix and the beginning of each submatrix
# GG: block-wise structure matrix 
# c: block-wise edge weight matrix

def aggregation(As: list, Cs: dict, GG:np.ndarray, size_arr:list, gamma:np.ndarray):
    layer = len(As)
    pos = np.cumsum(size_arr)
    size = sum(size_arr)
    blockmat = lil_matrix((size, size))
    for i in range(layer):
        for a, b in zip(As[i].nonzero()[0], As[i].nonzero()[1]):
            blockmat[pos[i]+a, pos[i]+b] = gamma[i,i] * As[i][a, b]
        for j in range(layer):
            if j > i and GG[i, j] != 0:
                for a, b in zip(Cs[(i, j)].nonzero()[0], Cs[(i, j)].nonzero()[1]):
                    x,y = pos[i]+a,pos[j]+b
                    blockmat[x, y] = gamma[i,j] * Cs[(i, j)][a, b]
                    blockmat[y, x] = gamma[j,i] * Cs[(i, j)][a, b]
    return blockmat, pos

# Convert the multi-layered network matrix to lil_matrix
def convertLil(As: list, Cs: dict, GG:np.ndarray):
    layer = len(As)
    for i in range(layer):
        As[i] = lil_matrix(As[i])
        for j in range(layer):
            if GG[i,j] != 0:
                Cs[(i,j)] = lil_matrix(Cs[(i,j)])

# Convert the multi-layered network matrix to csc_matrix
def convertCsc(As: list, Cs: dict, GG: np.ndarray):
    layer = len(As)
    for i in range(layer):
        As[i] = csc_matrix(As[i])
        for j in range(layer):
            if GG[i, j] != 0:
                Cs[(i, j)] = csc_matrix(Cs[(i, j)])

def convert_tolil(As:list,Cs:dict):
    within = [lil_matrix(a) for a in As]
    cross = {key:lil_matrix(Cs[key]) for key in Cs.keys()}
    return within,cross

def convert_tocsc(As:list,Cs:dict):
    within = [csc_matrix(a) for a in As]
    cross = {key:csc_matrix(Cs[key]) for key in Cs.keys()}
    return within,cross

# Sample a sub multi-layered network via candidate nodes
def sampleML(As: list, Cs: dict, GG:np.ndarray,candidate):
    layer = len(As)
    tmpAs = []
    tmpCs = dict()
    for i in range(layer):
        tmpAs.append(As[i][candidate[i], :][:, candidate[i]])
        for j in range(layer):
            if j > i and GG[i, j] != 0:
                tmpCs[(i, j)] = Cs[(i, j)][candidate[i], :][:, candidate[j]]
                tmpCs[(j, i)] = tmpCs[(i, j)].T
    return tmpAs,tmpCs

# Merge the multi-layered network into one block matrix.
def mergeML(As:list,Cs:dict,GG:np.ndarray,gamma:np.ndarray):
    layer = len(As)
    candidate = [list(range(a.shape[0])) for a in As]
    size_arr = [0] + [len(cand) for cand in candidate]
    totalMat,_ = aggregation(As, Cs, GG, size_arr, gamma)
    return totalMat

def indicator(res:list):
    for i in range(len(res)):
        if len(res[i]) == 0:
            return 0
    return 1


""" Parameters in intduen
    As : list of within-layer adjacency matrix 
    Cs : dict of cross-layer dependency matrix
    GG : the structure of the dependency in multi-layered network
    gamma: edge importance for cross-layer network
    boost: Optional, boosting or not. (Note: we do not recommend boosting for network with huge NNZ.)
    beta : coupled strength in matrix factorization
    R    : low-rank
    epoch: iteration number
    reg  : l1 regularization
    seed : random seed
"""

def Intduen(As: list, Cs: dict, GG: np.ndarray, gamma:np.ndarray, boost=True, beta=None, R=10, epoch=50, reg=1e-10, seed=1234):

    convertCsc(As,Cs,GG)
    U, lamb, sigm, beta = CONF(As, Cs, GG, beta,R,epoch,reg,seed)
    convertLil(As,Cs,GG)

    layer = len(As)
    nsize = sum([factor.shape[0] for factor in U])

    bestRes   = None
    bestScore = 0
    globalRes = []

    if boost:
        totalCandidate = [list(range(graph.shape[0])) for graph in As]
        total_size_arr = [0] + [len(layerCand) for layerCand in totalCandidate]
        # Merge the multi-layered network into a large network
        totalMat, totalPos = aggregation(As, Cs, GG, total_size_arr, gamma)

    

    for r in range(R):
        candidate = [None] * layer
        for i in range(layer):

            # Select the candidate nodes for the i-th layer
            delta = np.sqrt(1/U[i].shape[0])                        # select the nodes with score larger than delta
            topk  = int(0.02*U[i].shape[0]*(1-U[i].shape[0]/nsize)) # select the top 1% nodes based on the score
            cand_layer_delta = sorted(np.argwhere(U[i][:, r] > delta)[:, 0])
            cand_layer_topk  = sorted(np.argpartition(U[i][:,r],-1*topk)[-1*topk:])
            candidate[i] = cand_layer_delta if len(cand_layer_delta)>=len(cand_layer_topk) else cand_layer_topk

        # Construct the multi-layered subgraph based on the candidate nodes
        tmpAs,tmpCs = sampleML(As,Cs,GG,candidate)
        # Aggregation the multi-layered subgraph into a block matrix 
        size_arr = [0] + [len(cand) for cand in candidate]
        blockMat, pos = aggregation(tmpAs, tmpCs, GG, size_arr, gamma)  
        density = blockMat.sum() / (2 * blockMat.shape[0])
        print("After candidating, size of each layer: ", [len(cand) for cand in candidate])
        print("The joint density is {:.4f}".format(density))

        # The greedy_charikar can be replaced by other DSD solvers.
        res,density = greedyCharikar(blockMat)  
        res = np.array(sorted(list(res)))

        # Split the result into the corresponding nodes in each layer
        layerRes = []
        for i in range(layer):
            tmp_res = res[(res >= pos[i]) & (res < [pos[i+1]])] - pos[i]
            tmp_layer_res = [candidate[i][idx] for idx in list(tmp_res)]
            layerRes.append(tmp_layer_res)

        print("After dsd detector, size of each layer: ", [len(re) for re in layerRes])
        print("The joint density is {:.4f}".format(density))

        # Boosting for each column vector group.
        if boost:
            totalRes = []
            for i in range(layer):
                totalRes.extend([idx+totalPos[i] for idx in layerRes[i]])
            totalRes, density = fastExpander(totalMat, totalRes, density)
            totalRes = np.array(sorted(list(totalRes)))
            # Split the totalRes into each layer
            for i in range(layer):
                layerRes[i] = list(totalRes[(totalRes >= totalPos[i]) & (
                    totalRes < [totalPos[i+1]])] - totalPos[i])
            
            print("After neighbor boosting, size of each layer: ", [len(re) for re in layerRes])
            print("The joint density is {:.4f}".format(density))

        
        density *= indicator(layerRes)    

        print("The optimal joint density of the {}-th column of factors: {:.4f}".format(r,density))
        print('\n')

        if density > bestScore:
            bestScore = density
            bestRes = layerRes
        globalRes.append((layerRes,density))

    return bestRes,bestScore,globalRes
