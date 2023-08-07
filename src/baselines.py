import sys
sys.path.append("..")

import time
import numpy as np
from cmath import inf

from src.MinTree import MinTree
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
from scipy.sparse import linalg


""" Greedy algorithm for finding the densest subgraph
    graph : the input undirected and unweighted graph in scipy sparse matrix
"""

def greedyCharikar(graph:lil_matrix):
    graph = graph.tolil()    # the current graph adjacency matrix
    curScore = graph.sum() / 2
    Set = set(range(0, graph.shape[1]))
    bestAveScore = curScore / len(Set)
    deltas = np.squeeze(graph.sum(axis=1).A)
    tree = MinTree(deltas)
    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    while len(Set) > 1:
        node, val = tree.getMin()
        curScore -= val
        # Update priority for the neighbors of the deleted node
        for j in graph.rows[node]:
            delt = graph[node, j]
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
    finalSet = set(range(0, graph.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    finalres = sorted(finalSet)
    return finalres, bestAveScore

""" GreedyOQC algorithm for finding the alpha-quasi-clique
    graph : the input undirected and unweighted graph in scipy sparse matrix
    alpha : parameter of the alpha-quasi-clique, default 1/3
"""

def greedyOqc(graph:lil_matrix, alpha=1/3):
    graph = graph.tolil()    # the current graph adjacency matrix
    edges = graph.sum() / 2   # sum of edges
    Set = set(range(0, graph.shape[1]))
    best_score = curScore = edges - alpha * (len(Set)*(len(Set)-1))/2
    deltas = np.squeeze(graph.sum(axis=1).A)
    tree = MinTree(deltas)
    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    while len(Set) > 2:
        node, val = tree.getMin()
        edges -= val
        # Update priority for the neighbors of the deleted node
        for j in graph.rows[node]:
            delt = graph[node, j]
            tree.changeVal(j, -delt)
        Set -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        numDeleted += 1
        curScore = edges - alpha * (len(Set)*(len(Set)-1))/2
        if curScore > best_score:
            best_score = curScore
            bestNumDeleted = numDeleted
    # reconstruct the best sets
    finalSet = set(range(0, graph.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    finalres = sorted(finalSet)
    return finalres, best_score


def specgreedy(graph:lil_matrix):
    from specgreedy.main import specgreedy_monopartite
    res,score = specgreedy_monopartite(graph)
    return res,score
    
def coreds(graph:lil_matrix):
    from eds_kcore.main import efficient_core_dsd
    res,score = efficient_core_dsd(graph)
    return res,score

""" The MF-DSD method described as algorithm 2 in our paper
    which is similar with the eigenspokes algorithm.
    graph :  the input undirected and unweighted graph in scipy lil_matrix
    factor : the factor matrix obtained from nmf,svd,nncf or other matrix factorization techniques
    metric: average density(default), volume density or others
"""

def mfdsd(graph:lil_matrix, k=10, metric='average'):
    RU, RS, RVt = linalg.svds(graph.asfptype(), k)
    best_score = -inf
    number_of_nodes, rank = RU.shape
    noru = normalize(RU, axis=0)  # column vector normalization
    threshold = np.sqrt(1/number_of_nodes)
    for idx in range(rank):
        candidate = list(np.argwhere(np.abs(noru[:, idx]) > threshold)[:, 0])
        subgraph = graph.asfptype()[candidate, :][:, candidate]
        if len(candidate) > 1:
            if metric == 'average':
                score = subgraph.sum() / len(candidate)
            elif metric == 'clique':
                score = subgraph.sum() / (len(candidate)*(len(candidate)-1))
            if score > best_score:
                best_score = score
                finalres = candidate.copy()
        else:
            raise ValueError("This candidates has no more than 1 element")
    return finalres, best_score