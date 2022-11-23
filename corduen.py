import numpy as np
from baselines import greedyCharikar
from scipy.sparse import lil_matrix

from boosting import neighborBoosting
from nncf import NNCF


# return the block matrix and the beginning of each submatrix
def aggregation(As: list, Cs: dict, GG, size_arr, c):
    pos = np.cumsum(size_arr)
    size = sum(size_arr)
    blockmat = lil_matrix((size, size))
    layer = len(As)
    for i in range(layer):
        for a, b in zip(As[i].nonzero()[0], As[i].nonzero()[1]):
            blockmat[pos[i]+a, pos[i]+b] = As[i][a, b]
        for j in range(layer):
            if j > i and GG[i, j] != 0:
                for a, b in zip(Cs[(i, j)].nonzero()[0], Cs[(i, j)].nonzero()[1]):
                    blockmat[pos[i]+a, pos[j]+b] = c * Cs[(i, j)][a, b]
                    blockmat[pos[j]+b, pos[i]+a] = c * Cs[(i, j)][a, b]
    return blockmat, pos


def Corduen(As: list, Cs: dict, GG, ra=10, boost=True, R=10, epoch=50, reg=1e-6, seed=1234):

    # NNCF to get the factor for each layer
    U, lamb, sigm, alpha = NNCF(As, Cs, GG, epoch, R, reg, seed)

    total_candidate = [list(range(a.shape[0])) for a in As]
    total_size_arr = [0] + [len(layer_cand) for layer_cand in total_candidate]
    # which could take a long time.
    totalMat, pos = aggregation(As, Cs, GG, total_size_arr, ra)
    total_pos = np.cumsum(total_size_arr)

    layer = len(As)
    for idx in range(layer):
        As[idx] = lil_matrix(As[idx])

    boost_finalres = None
    bestScore = 0
    for r in range(R):
        candidate = []
        for i in range(layer):
            delta = np.sqrt(1/U[i].shape[0])
            u = list(np.argwhere(np.array(U[i][:, r]) > delta)[:, 0])
            candidate.append(u)

        # produce temp As and Cs
        tmpAs = []
        tmpCs = dict()
        for i in range(layer):
            tmpAs.append(As[i][candidate[i], :][:, candidate[i]])
            for j in range(layer):
                if j > i and GG[i, j] != 0:
                    tmpCs[(i, j)] = Cs[(i, j)][candidate[i], :][:, candidate[j]]
                    tmpCs[(j, i)] = tmpCs[(i, j)].T

        size_arr = [0] + [len(layer_cand) for layer_cand in candidate]
        blockMat, pos = aggregation(
            tmpAs, tmpCs, GG, size_arr, ra)  # aggregation
        res, score = greedyCharikar(blockMat)  # greedy peeling
        res = np.array(sorted(list(res)))

        # Split the result for each layer
        layer_res = []
        for i in range(layer):
            tmp_res = res[(res >= pos[i]) & (res < [pos[i+1]])] - pos[i]
            tmp_layer_res = [candidate[i][idx] for idx in list(tmp_res)]
            layer_res.append(tmp_layer_res)

        total_res = []
        for i in range(layer):
            total_res.extend([idx+total_pos[i] for idx in layer_res[i]])

        # We don't recommend boosting for lagre-scale multi-layered networks.
        if boost:
            total_res, score = neighborBoosting(totalMat, total_res)

        print(f"{r}-th column vector group , score {score}")

        if score > bestScore:
            bestScore = score
            boost_finalres = total_res

        boost_finalres = np.array(sorted(boost_finalres))

    # Split the boost_finalres into each layer
    finalres = []
    for i in range(layer):
        tmp_finalres = boost_finalres[(boost_finalres >= total_pos[i]) & (
            boost_finalres < [total_pos[i+1]])] - total_pos[i]
        finalres.append(tmp_finalres)

    return finalres
