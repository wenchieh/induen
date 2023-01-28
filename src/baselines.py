from cmath import inf
import numpy as np
from src.MinTree import MinTree
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix


""" Greedy algorithm for finding the densest subgraph
    graph : the input undirected and unweighted graph in scipy sparse matrix
"""

def greedyCharikar(graph):
    Mcur = graph.tolil()    # the current graph adjacency matrix
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
        # Update priority for the neighbors of the deleted node
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


""" GreedyOQC algorithm for finding the alpha-quasi-clique
    graph : the input undirected and unweighted graph in scipy sparse matrix
    alpha : parameter of the alpha-quasi-clique, default 1/3
"""


def greedyOqc(graph, alpha=1/3):
    Mcur = graph.tolil()    # the current graph adjacency matrix
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
        # Update priority for the neighbors of the deleted node
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
    return finalSet, bestScore

""" The MF-DSD method described as algorithm 2 in our paper
    graph :  the input undirected and unweighted graph in scipy lil_matrix
    factor : the factor matrix obtained from nmf,svd,nncf or other matrix factorization techniques
    metric: average density(default), volume density or others
"""


def mfdsd(graph: lil_matrix, factor, metric='average'):
    bestScore = -inf
    nodenum, rank = factor.shape
    noru = normalize(factor, axis=0)  # column vector normalization
    threshold = np.sqrt(1/nodenum)
    for idx in range(rank):
        # select import nodes based on the threshold sqrt(1/n)
        candidate = list(np.argwhere(np.abs(noru[:, idx]) > threshold)[:, 0])
        # construct subgraph by the candidate node list
        subgraph = graph.asfptype()[candidate, :][:, candidate]
        if len(candidate) > 1:
            if metric == 'average':
                score = subgraph.sum() / len(candidate)
                # finalSet, score = fastGreedyDecreasing4DSD(graph)
            elif metric == 'clique':
                score = subgraph.sum() / (len(candidate)*(len(candidate)-1))
                # finalSet, score = fastGreedyDecreasing(graph)
            if score > bestScore:
                bestScore = score
                finalres = candidate.copy()
        else:
            raise ValueError(
                "This candidate noedlist has no more than 1 element")
    return finalres, bestScore
