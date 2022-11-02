from calendar import prweek
from cmath import inf
import os
import time
import pickle
import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.io as scio
import networkx as nx
import matplotlib.pyplot as plt

from math import sqrt
from MinTree import MinTree
from numpy import matmul
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd, squared_norm
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


# Norm for both dense and sparse matrix
def norm(x):
    if sp.issparse(x):
        return sp.linalg.norm(x)
    return np.linalg.norm(x)

# Check a matrix symmetric or not


def issymmetric(x):
    if x.shape[0] != x.shape[1]:
        return False
    flag = False
    if sp.issparse(x):
        flag = bool(1-(x.T-x).nnz)
    return flag

# Check a graph have self edges or not


def ishaveSelfEdge(g):
    # g : sparse adjacency matrix
    for idx in range(g.shape[0]):
        if g[idx, idx]:
            return True
    return False

# Create a random matrix with size and probability


def random_matrix(size, prob):
    a = np.random.random(size)
    return np.where(a < prob, 1, 0)

# Error for model the two coupled matrix


def error(G1, G2, C, U, V, W, alpha):
    return squared_norm(G1-U) + squared_norm(G2-V) + alpha * squared_norm(C-W)

# Calculate he gradient of U


def gradient_U(U, V, G1, C, alpha):
    nabla_f_u = matmul(matmul(U, V.T)-C, V)
    nabla_g_u = 2 * matmul(matmul(U, U.T)-G1, U)
    return alpha * nabla_f_u + nabla_g_u

# Calculate the gradient of Vt


def gradient_V(U, V, G2, C, alpha):
    nabla_f_v = matmul(matmul(U.T, U)-C.T, U)
    nabla_g_v = 2 * matmul(matmul(V.T, V)-G2, V)
    return alpha * nabla_f_v + nabla_g_v

# Update U,V via Adam Optimizer


def AdamUpdate(G1, G2, C, R, epochs, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=0):

    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    U = np.random.rand(G1.shape[0], R)
    V = np.random.rand(G2.shape[0], R)
    total_error = [error(G1, G2, C, matmul(
        U, U.T), matmul(V, V.T), matmul(U, V.T), alpha)]
    mt = vt = vt_max = np.zeros(U.shape, dtype=np.float64)
    pt = qt = qt_max = np.zeros(V.shape, dtype=np.float64)
    for t in range(1, epochs+1):
        # Update U
        gt = gradient_U(U, V, G1, C, alpha)
        # gt = gt / norm(gt) # normalization
        gt[np.isnan(gt)] = 0.01
        mt = betas[0] * mt + (1-betas[0]) * gt
        vt = betas[1] * vt + (1-betas[1]) * np.power(gt, 2)
        mt_hat = mt / (1-betas[0] ** t)
        vt_hat = vt / (1-betas[1] ** t)
        vt_max = np.maximum(vt_hat, vt_max)
        if amsgrad:
            U -= weight_decay*U + lr * np.divide(mt_hat, np.sqrt(vt_max)+eps)
        else:
            U -= weight_decay*U + lr * np.divide(mt_hat, np.sqrt(vt_hat)+eps)
        # Update V
        ht = gradient_V(U, V, G2, C, alpha)
        # ht = ht / norm(ht) # normalization
        ht[np.isnan(ht)] = 0.01
        pt = betas[0] * pt + (1-betas[0]) * ht
        qt = betas[1] * qt + (1-betas[1]) * np.power(ht, 2)
        pt_hat = pt / (1-betas[0] ** t)
        qt_hat = qt / (1-betas[1] ** t)
        qt_max = np.maximum(qt_hat, qt_max)
        if amsgrad:
            V -= weight_decay * V + lr * \
                np.divide(pt_hat, np.sqrt(qt_max) + eps)
        else:
            V -= weight_decay * V + lr * \
                np.divide(pt_hat, np.sqrt(qt_hat) + eps)
        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(U, U.T),
                                 matmul(V, V.T), matmul(U, V.T), alpha))
    return U, V, total_error

# Gradient descent update


def GradientUpdate(G1, G2, C, R, epochs, lr=1e-4, weight_decay=0.01):
    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    U = np.random.rand(C.shape[0], R)
    V = np.random.rand(C.shape[1], R)
    total_error = [error(G1, G2, C, matmul(
        U, U.T), matmul(V, V.T), matmul(U, V.T), alpha)]
    for iter in range(epochs):
        # Update U
        nabla_u = gradient_U(U, V, G1, C, alpha)
        nabla_u = nabla_u / norm(nabla_u)
        nabla_u[np.isnan(nabla_u)] = 0.01
        U -= weight_decay * U + lr * nabla_u
        # Update V
        nabla_v = gradient_V(U, V, G2, C, alpha)
        nabla_v = nabla_v / norm(nabla_v)
        nabla_v[np.isnan(nabla_v)] = 0.01
        V -= weight_decay * V + lr * nabla_v
        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(U, U.T),
                                 matmul(V, V.T), matmul(U, V.T), alpha))
    return U, V, total_error


# Multiplicative update of coupled matrix factorization
# Minimize the objective Fro(G1-u@u.T) + Fro(G2-v@v.T) + alpha*Fro(C-u@v.T)
# where Fro means the Frobenius norm of the matrix
def MU(G1, G2, C, R, epochs):
    # G1,G2 : lil_matrix which represent two layers
    # C : the cross-layer dependency matrix

    # Initialize U,V,alpha
    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    u = np.random.rand(G1.shape[0], R)
    v = np.random.rand(G2.shape[1], R)

    # Error of initialization
    total_error = [error(G1, G2, C, matmul(
        u, u.T), matmul(v, v.T), matmul(u, v.T), alpha)]

    for it in range(epochs):
        # Update U
        u_upper = 2 * G1 * u + alpha * matmul(C, v)
        u_lower = matmul(u, 2 * matmul(u.T, u) + alpha * matmul(v.T, v))
        u_res = np.power(np.divide(u_upper, u_lower), 1/2)
        u_res[np.isnan(u_res)] = 0.01
        u = np.multiply(u, u_res)

        # Update V
        v_upper = 2 * G2 * v + alpha * matmul(C.T, u)
        v_lower = matmul(v, 2 * matmul(v.T, v) + alpha * matmul(u.T, u))
        v_res = np.power(np.divide(v_upper, v_lower), 1/2)
        v_res[np.isnan(v_res)] = 0.01
        v = np.multiply(v, v_res)

        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(u, u.T),
                                 matmul(v, v.T), matmul(u, v.T), alpha))
    return u, v, total_error


#  Multi-multiplicative updates of coupled matrix factorization
def MMU(G1, G2, G3, C12, C13, C23, R, epochs):
    alpha = (norm(G1)+norm(G2))/(2*norm(C12))
    beta = (norm(G1)+norm(G3))/(2*norm(C13))
    gamma = (norm(G2)+norm(G3))/(2*norm(C23))

    U = np.random.rand(G1.shape[0], R)
    V = np.random.rand(G2.shape[0], R)
    W = np.random.rand(G3.shape[0], R)
    # Error of initialization
    total_error = [squared_norm(G1-U@U.T)+squared_norm(G2-V@V.T)+squared_norm(G3-W@W.T) +
                   alpha*squared_norm(C12-U@V.T)+beta*squared_norm(C13-U@W.T)+gamma*squared_norm(C23-V@W.T)]

    for iter in range(epochs):
        # Update U
        u_upper = 2 * G1 @ U + alpha*C12@V + beta*C13@W
        u_lower = U @ (2*U.T@U + alpha*V.T@V + beta*W.T@W)
        u_res = np.power(np.divide(u_upper, u_lower), 1/2)
        u_res[np.isnan(u_res)] = 0.01
        U = np.multiply(U, u_res)
        # Update V
        v_upper = 2*G2@V + alpha*C12.T@U + gamma*C23@W
        v_lower = V @ (2*V.T@V + alpha*U.T@U + gamma*W.T@W)
        v_res = np.power(np.divide(v_upper, v_lower), 1/2)
        v_res[np.isnan(v_res)] = 0.01
        V = np.multiply(V, v_res)
        # Update W
        w_upper = 2*G3@W + beta*C13.T@U + gamma*C23.T@V
        w_lower = W @ (2*W.T@W + beta*U.T@U + gamma*V.T@V)
        w_res = np.power(np.divide(w_upper, w_lower), 1/2)
        w_res[np.isnan(w_res)] = 0.01
        W = np.multiply(W, w_res)
        # Calculate the reconstruction error
        total_error.append(squared_norm(G1-U@U.T)+squared_norm(G2-V@V.T)+squared_norm(G3-W@W.T) +
                           alpha*squared_norm(C12-U@V.T)+beta*squared_norm(C13-U@W.T)+gamma*squared_norm(C23-V@W.T))
    return U, V, W, total_error


def Greedy(G, U):
    Score = []
    selector = []
    for idx in range(U.shape[1]):
        bestSet = set()
        delta = np.sqrt(squared_norm(U[:, idx]) / U.shape[0])
        candidateSet = np.argwhere(np.array(U[:, idx]) > delta)[:, 0]
        # print("Length of candidate set", len(candidateSet))
        g = G.copy()
        g = g[candidateSet, :][:, candidateSet]
        degree = np.squeeze(g.sum(axis=0).A)          # d: degree array
        # Initialize the set to start remove greedily
        curSet = set(range(len(candidateSet)))
        tree = MinTree(degree)
        curScore = sum(degree) / 2
        bestAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))

        while len(curSet) > 2:
            node, val = tree.getMin()
            # node 是tree输入数组中最小值元素的索引
            curSet -= {node}
            curScore -= val
            # print('max_density',max_density,'node',node,'len s',len(curSet))
            tree.setVal(node, float('inf'))
            # Update priority of neighbors
            for j in g.rows[node]:
                delt = g[node, j]
                tree.changeVal(j, -delt)
            g[node, :], g[:, node] = 0, 0
            curAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))
            if curAveScore > bestAveScore:
                bestAveScore = curAveScore
                bestSet = curSet.copy()
        pos = list(bestSet)
        res = list(np.array(candidateSet)[pos])
        Score.append(bestAveScore)
        selector.append(res)
        print("Maximum density:", bestAveScore,
              "len of optimal set:", len(bestSet))
    return Score, selector


# Greedy search for find a dense subgraph in a monopartite graph.
# Input : sparse graph adjacency matrix
# Output : subgraph and its corresponding score
def fastGreedyDecreasing(G):
    # Mcur is a sysmmetric matrix.
    # Mcur : lil_matrix
    Mcur = G.tolil()
    curScore = Mcur.sum() / 2
    Set = set(range(0, Mcur.shape[1]))
    bestAveScore = 2 * curScore / (len(Set)*(len(Set)-1))
    Deltas = np.squeeze(Mcur.sum(axis=1).A)
    tree = MinTree(Deltas)
    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    while len(Set) > 2:
        node, val = tree.getMin()
        curScore -= val
        # Update priority
        for j in Mcur.rows[node]:
            delt = Mcur[node, j]
            tree.changeVal(j, -delt)
        Set -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        numDeleted += 1
        curAveScore = 2 * curScore / (len(Set)*(len(Set)-1))
        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted
    # reconstruct the best sets
    finalSet = set(range(0, Mcur.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    return finalSet, bestAveScore


def mfdsd(g: lil_matrix, u, metric='average'):
    """The MF-DSD method described as algorithm 2 in our paper
    g : the adjacency matrix
    u : the factor matrix
    metric: 'average' or 'clique'
    """
    bestScore = -inf
    nodenum, rank = u.shape
    noru = normalize(u, axis=0)  # column vector normalization
    threshold = np.sqrt(1/nodenum)
    for idx in range(rank):
        # select import nodes based on the threshold
        candidate = list(np.argwhere(np.abs(noru[:, idx]) > threshold)[:, 0])
        # subgraph construed via the candidate list: lil_matrix
        graph = g.asfptype()[candidate, :][:, candidate]
        if len(candidate) > 1:
            if metric == 'average':
                score = graph.sum() / len(candidate)
                # finalSet, score = fastGreedyDecreasing4DSD(graph)
            elif metric == 'clique':
                score = graph.sum() / (len(candidate)*(len(candidate)-1))
                # finalSet, score = fastGreedyDecreasing(graph)
            if score > bestScore:
                bestScore = score
                finalres = candidate.copy()
        else:
            raise ValueError("The candidate set has no more than 1 element")
    return finalres, bestScore


def greedy(G, U, metric='average'):
    # G : the adjacency matrix
    # U : the factor matrix
    # metric: 'average' or 'clique'
    bestScore = []
    res = []
    nor_u = normalize(U, axis=0)
    delta = np.sqrt(1/U.shape[0])
    for idx in range(U.shape[1]):
        Set = list(np.argwhere(np.array(nor_u[:, idx]) > delta)[:, 0])
        g = G.copy().asfptype()
        g = g[Set, :][:, Set]
        if len(Set) > 1:
            if metric == 'average':
                finalSet, score = fastGreedyDecreasing4DSD(g)
            elif metric == 'clique':
                finalSet, score = fastGreedyDecreasing(g)
            print('length of bestSet', len(finalSet), 'bestScore', score)
            pos = list(finalSet)
            tmp_res = list(np.array(Set)[pos])
            bestScore.append(score)
            res.append(tmp_res)
    return res, bestScore

# Give a graph and its indicator vector ,calculate the score.


def checkScore(G, selector):
    # G : lil_matrix
    # selector: array_like
    score = G[selector, :][:, selector].sum()
    if len(selector) == 0 or len(selector) == 1:
        aveScore = 0
    else:
        aveScore = score / (len(selector) * (len(selector)-1))
    return aveScore


def metric(A: lil_matrix, S: set, description):
    # You can define novel metric by yourself
    # A is symmetric adjacency matrix by default
    edges = A[list(S), :][:, list(S)].sum() / 2
    if description == 'clique':
        res = 2 * edges / (len(S)*(len(S)-1))
    elif description == 'average':
        res = edges / len(S)


# Give a graph and its indicator vector, extend the subgraph with higher score.
def extend(g: lil_matrix, sub, description='average'):
    # g:lil_matrix
    # sub: which means subgraph that to be extend , array_like data
    # curScore : number of edges of sum of degrees
    subgraph = sub.copy()
    curScore = g[subgraph, :][:, subgraph].sum() / 2
    # curAveScore = metric(g,set(sub),description)
    curAveScore = 2 * curScore / (len(subgraph)*(len(subgraph)-1))
    curSet = set(subgraph)
    # the nodes added to the subgraph
    nodes = []
    addSet = set()
    # We just simply check the neighbors of the nodes that not in the current subgraph
    # Or we can get the top_k neighbors with high degrees as the candidate set.
    for node in subgraph:
        subSet = set(g.rows[node]) - curSet
        addSet = addSet | subSet
    for neigh in addSet:
        addedges = g[neigh][:, subgraph].sum()
        # tmpAveScore = 2 * (curScore+addedges) / (len(curSet)*(len(curSet)+1))
        tmpAveScore = 2 * (curScore+addedges) / (len(curSet)*(len(curSet)+1))
        if tmpAveScore >= curAveScore:
            curScore += addedges
            curAveScore = tmpAveScore
            curSet.add(neigh)
            subgraph.append(neigh)
            nodes.append(neigh)
            tmp_addSet = set(g.rows[neigh]) - curSet
            addSet = addSet | tmp_addSet
    if len(nodes):
        print("Add nodes:", nodes, 'length of current subgraph:',
              len(subgraph), "curAveScore:", curAveScore)
    else:
        print("No nodes add to the current subgraph!")
    return subgraph, curAveScore


def TriMU(G: list, C: list, R, epochs, reg=1e-6):
    """G:list of adjacency matrix
       C:list of cross-layer dependency matrix
    """
    g1, g2, g3 = G[0], G[1], G[2]
    c12, c13, c23 = C[0], C[1], C[2]
    alpha = (norm(g1)+norm(g2)) / (2*norm(c12))
    beta = (norm(g1)+norm(g3)) / (2*norm(c13))
    gamma = (norm(g2)+norm(g3)) / (2*norm(c23))
    # Initialize U,V,W
    U = np.random.rand(g1.shape[0], R)
    V = np.random.rand(g2.shape[0], R)
    W = np.random.rand(g3.shape[0], R)

    # L1-norm　regularization term
    reg_u = reg * np.ones(U.shape)
    reg_v = reg * np.ones(V.shape)
    reg_w = reg * np.ones(W.shape)

    # Multiplicative Update
    for it in range(epochs):
        # Update U
        u_upper = 2 * g1 * U + alpha * c12 @ V + beta * c13 @ W
        u_lower = U @ (2*U.T@U + alpha*V.T@V + beta*W.T@W) + reg_u
        u_res = np.power(np.divide(u_upper, u_lower), 1/2)
        # u_res[np.isnan(u_res)] = 0.01
        U = np.multiply(U, u_res)

        # Update V
        v_upper = 2 * g2 * V + alpha * c12.T @ U + gamma * c23 @ W
        v_lower = V @ (2*V.T@V + alpha*U.T@U + gamma*W.T@W) + reg_v
        v_res = np.power(np.divide(v_upper, v_lower), 1/2)
        # v_res[np.isnan(v_res)] = 0.01
        V = np.multiply(V, v_res)

        # Update W
        w_upper = 2 * g3 * W + beta * c13.T @ U + gamma * c23.T @ V
        w_lower = W @ (2*W.T@W + beta*U.T@U + gamma*V.T@V) + reg_w
        w_res = np.power(np.divide(w_upper, w_lower), 1/2)
        # w_res[np.isnan(w_res)] = 0.01
        W = np.multiply(W, w_res)
    return U, V, W


def fastGreedyDecreasing4DSD(G):
    # Mcur is a sysmmetric matrix.
    # Mcur : lil_matrix
    Mcur = G.tolil()
    curScore = Mcur.sum() / 2
    Set = set(range(0, Mcur.shape[1]))
    bestAveScore = curScore / len(Set)
    Deltas = np.squeeze(Mcur.sum(axis=1).A)
    tree = MinTree(Deltas)
    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    while len(Set) > 1:
        node, val = tree.getMin()
        curScore -= val
        # Update priority
        for j in Mcur.rows[node]:
            delt = Mcur[node, j]
            tree.changeVal(j, -delt)
        Set -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        numDeleted += 1
        curAveScore = curScore / len(Set)
        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted
    # reconstruct the best sets
    finalSet = set(range(0, Mcur.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    return finalSet, bestAveScore


# 二部图贪心删除算法，用于寻找biclique
def bifastGreedyDecreasing(M):
    (m, n) = M.shape
    Ml = M.tolil()
    Mlt = M.transpose().tolil()
    rowSet = set(range(0, m))
    colSet = set(range(0, n))
    curScore = M[list(rowSet), :][:, list(colSet)].sum(axis=None)

    bestAveScore = curScore / (len(rowSet) + len(colSet))
    bestSets = (rowSet, colSet)
    # print("finished setting up greedy")
    # *decrease* in total weight when *removing* this row
    # Prepare the min priority tree to begin greedy algorithm.
    rowDeltas = np.squeeze(M.sum(axis=1).A)
    colDeltas = np.squeeze(M.sum(axis=0).A)
    # print("finished setting deltas")
    rowTree = MinTree(rowDeltas)
    colTree = MinTree(colDeltas)
    # print("finished building min trees")

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    while rowSet and colSet:
        if (len(colSet) + len(rowSet)) % 100000 == 0:
            print("current set size = ", len(colSet) + len(rowSet))
        nextRow, rowDelt = rowTree.getMin()
        nextCol, colDelt = colTree.getMin()

        if rowDelt <= colDelt:
            curScore -= rowDelt
            # Update priority for the node with min priority and its neighbors
            for j in Ml.rows[nextRow]:
                delt = Ml[nextRow, j]
                colTree.changeVal(j, -delt)
            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))
        else:
            curScore -= colDelt
            # Update priority for the node with min priority and its neighbors
            for i in Mlt.rows[nextCol]:
                delt = Ml[i, nextCol]
                rowTree.changeVal(i, -delt)
            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))

        numDeleted += 1
        curAveScore = curScore / (len(colSet) + len(rowSet))

        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted

    # Reconstruct the best row and column sets
    finalRowSet = set(range(m))
    finalColSet = set(range(n))
    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        else:
            finalColSet.remove(deleted[i][1])

    return (finalRowSet, finalColSet, bestAveScore)


def mkdir_self(dir):
    if dir.endswith('.txt'):
        dir_list = dir.split('/')
        dir = '/'.join(dir_list[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)


def saveres2txt(res, outpath):
    mkdir_self(outpath)
    for i in range(len(res[0])):
        res[0][i] = list(res[0][i])
    with open(outpath, "a") as f:
        f.write('[')
        f.write('\n')
        for i in range(len(res[0])):
            for j in range(len(res[0][i])):
                f.write(str(res[0][i][j]))
                f.write('\t')
            f.write('\n')
        f.write(str(res[1]))
        f.write('\n')
        f.write(']')
        f.write('\n')
    print('save res.txt success!')


# return the block matrix and the beginning of each submatrix
def aggregation(As: list, Cs: dict, GG, candidate: list, c):
    size_arr = [0] + [len(lis) for lis in candidate]
    pos = np.cumsum(size_arr)
    size = sum(size_arr)
    blockmat = lil_matrix((size, size))
    for i in range(len(As)):
        blockmat[pos[i]:pos[i+1], pos[i]:pos[i+1]
                 ] = As[i].tolil()[candidate[i], :][:, candidate[i]]
        for j in range(len(As)):
            if j > i and GG[i, j] != 0:
                blockmat[pos[i]:pos[i+1], pos[j]:pos[j+1]] = c * \
                    Cs[(i, j)][candidate[i], :][:, candidate[j]]
                blockmat[pos[j]:pos[j+1], pos[i]:pos[i+1]] = c * \
                    Cs[(i, j)][candidate[i], :][:, candidate[j]].T
    return blockmat, pos


def evaluate(u, v, w, truth_u, truth_v, truth_w, verbose=True):
    # Input : set
    tpu = u.intersection(truth_u)
    tpv = v.intersection(truth_v)
    tpw = w.intersection(truth_w)

    pre_u = 0 if len(u) == 0 else len(tpu) / len(u)
    pre_v = 0 if len(v) == 0 else len(tpv) / len(v)
    pre_w = 0 if len(w) == 0 else len(tpw) / len(w)

    rec_u = len(tpu) / len(truth_u)
    rec_v = len(tpv) / len(truth_v)
    rec_w = len(tpw) / len(truth_w)

    f1_u = 0 if (pre_u + rec_u) == 0 else 2 * \
        pre_u * rec_u / (pre_u + rec_u)
    f1_v = 0 if (pre_v + rec_v) == 0 else 2 * \
        pre_v * rec_v / (pre_v + rec_v)
    f1_w = 0 if (pre_w + rec_w) == 0 else 2 * \
        pre_w * rec_w / (pre_w + rec_w)

    if verbose:
        print("precision_u {:.3f}".format(
            pre_u),  "recall_u {:.3f}".format(rec_u), "F1_u {:.3f}".format(f1_u))
        print("precision_v {:.3f}".format(
            pre_v),  "recall_v {:.3f}".format(rec_v), "F1_v {:.3f}".format(f1_v))
        print("precision_w {:.3f}".format(
            pre_w),  "recall_w {:.3f}".format(rec_w), "F1_w {:.3f}".format(f1_w))

    return pre_u, rec_u, f1_u, pre_v, rec_v, f1_v, pre_w, rec_w, f1_w


def greedyOQC(M, alpha=1/3):
    # Mcur : lil_matrix and sysmmetric matrix.
    # alpha : positive number belong to (0,1)
    Mcur = M.tolil()
    edges = Mcur.sum() / 2   # sum of edges
    Set = set(range(0, Mcur.shape[1]))
    bestScore = curScore = edges - alpha * (len(Set)*(len(Set)-1))/2
    deltas = np.squeeze(Mcur.sum(axis=1).A)
    tree = MinTree(deltas)
    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    while len(Set) > 2:
        node, val = tree.getMin()
        edges -= val
        # Update priority
        for j in Mcur.rows[node]:
            delt = Mcur[node, j]
            tree.changeVal(j, -delt)
        Set -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        numDeleted += 1
        curScore = edges - alpha * (len(Set)*(len(Set)-1))/2
        if curScore > bestScore:
            bestScore = curScore
            bestNumDeleted = numDeleted
    # reconstruct the best sets
    finalSet = set(range(0, Mcur.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    density = Mcur[list(finalSet), :][:, list(finalSet)].sum(
    ) / (len(finalSet)*(len(finalSet)-1)) if len(finalSet) > 1 else 0
    return finalSet, bestScore, density
