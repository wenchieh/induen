import numpy as np
import time
from src.baselines import greedyCharikar
from scipy.sparse import lil_matrix,csc_matrix
from src.boosting import fastNeighborBoosting
from src.nncf import CNMF,CONMF


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

# 先不考虑Boosting,Boosting并不是一个本质的东西；
def CorduenWithoutMF(As: list, Cs: dict, GG, c=10):
    layer = len(As)
    candidate = [list(range(a.shape[0])) for a in As]
    size_arr = [0] + [len(layer_cand) for layer_cand in candidate]
    totalMat,pos = aggregation(As, Cs, GG, size_arr, c)
    res,score = greedyCharikar(totalMat) # peeling the total multi-layered network
    res = np.array(sorted(list(res)))
    finalres = []   # node subset for each layer
    for i in range(layer):
        tmp_res = res[(res>=pos[i]) & (res<[pos[i+1]])] - pos[i]
        finalres.append(tmp_res)
    return finalres,res,score


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


""" Parameters (beta,R,epoch,reg,seed) is introduced at CONMF
    Ortho: with orthogonal constraints or not
    boost: with boosting or not
"""
def Corduen(As: list, Cs: dict, GG: np.ndarray, gamma=np.ndarray, boost=False, Ortho=True, beta=None, R=10, epoch=50, reg=1e-10, seed=1234):
    

    # CONMF or CNMF to get the factor matrix for each layer
    convertCsc(As,Cs,GG)
    if Ortho:
        U, lamb, sigm, beta = CONMF(As, Cs, GG, beta,R,epoch,reg,seed)
    else:
        U, lamb, sigm, beta = CNMF(As, Cs, GG, beta,R,epoch,reg,seed)
    convertLil(As,Cs,GG)

    layer = len(As)
    bestRes   = None
    bestScore = 0
    globalRes = []

    if boost:
        totalCandidate = [list(range(graph.shape[0])) for graph in As]
        total_size_arr = [0] + [len(layerCand) for layerCand in totalCandidate]
        # Merge the multi-layered network into a large network
        totalMat, totalPos = aggregation(As, Cs, GG, total_size_arr, gamma)

    nsize = sum([factor.shape[0] for factor in U])

    for r in range(R):
        candidate = []
        for i in range(layer):
            # set the truncated threshold as sqrt(1/n) and the top-percent as 1%
            delta = np.sqrt(1/U[i].shape[0])
            # topk = int(0.01*U[i].shape[0])
            topk  = int(0.02*U[i].shape[0]*(1-U[i].shape[0]/nsize))
            uThreshold = list(np.argwhere(U[i][:, r] > delta)[:, 0])
            uPercent = list(np.argpartition(U[i][:,r],-1*topk)[-1*topk:])
            # Select the node subset with larger size.
            if len(uThreshold)<len(uPercent):
                candidate.append(uPercent)
            else:
                candidate.append(uThreshold) 

        print(f"After candidating, size: {[len(cand) for cand in candidate]}")

        # Construct the multi-layered subgraph based on the candidate nodes
        tmpAs,tmpCs = sampleML(As,Cs,GG,candidate)

        # Aggregation the multi-layered subgraph into a block matrix 
        size_arr = [0] + [len(cand) for cand in candidate]
        blockMat, pos = aggregation(tmpAs, tmpCs, GG, size_arr, gamma)  

        # The greedy peeling can be replaced by other DSD solvers.
        res, score = greedyCharikar(blockMat)  
        res = np.array(sorted(list(res)))

        # Split the result into corresponding nodes in each layer
        layerRes = []
        for i in range(layer):
            tmp_res = res[(res >= pos[i]) & (res < [pos[i+1]])] - pos[i]
            tmp_layer_res = [candidate[i][idx] for idx in list(tmp_res)]
            layerRes.append(tmp_layer_res)

        print(f"After greedy peeling, size: {[len(re) for re in layerRes]}, averaged degree density: {score}")

        # Boosting for each column vector group.
        if boost:
            totalRes = []
            for i in range(layer):
                totalRes.extend([idx+totalPos[i] for idx in layerRes[i]])
            totalRes, score = fastNeighborBoosting(totalMat, totalRes, score)
            totalRes = np.array(sorted(list(totalRes)))
            # Split the totalRes into each layer
            for i in range(layer):
                layerRes[i] = list(totalRes[(totalRes >= totalPos[i]) & (
                    totalRes < [totalPos[i+1]])] - totalPos[i])
            
            print(f"After neighbor boosting, size: {[len(re) for re in layerRes]}, averaged degree density: {score}")

        
        for i in range(layer):
            score = 0 if len(layerRes[i])==0 else score

        print(f"Correlated density of the {r}-th group of column vectors: {score}")
        print('\n')

        if score > bestScore:
            bestScore = score
            bestRes = layerRes.copy()
        globalRes.append((layerRes,score))

    return bestRes,bestScore,globalRes
