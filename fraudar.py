# From paper FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
# https://dl.acm.org/doi/10.1145/2939672.2939747

import numpy as np
from scipy.sparse import lil_matrix
from MinTree import MinTree


def logWeightedAveDegree(self, graph):
    # M: scipy sparse matrix
    (m, n) = graph.shape
    colSums = graph.sum(axis=0)
    colWeights = 1.0 / np.log(np.squeeze(colSums.A) + 5)
    colDiag = lil_matrix((n, n))
    colDiag.setdiag(colWeights)
    return graph * colDiag


def Fraudar(graph):
    Mcur = graph.copy().tolil()
    weight_matrix = logWeightedAveDegree(Mcur)
    return greedyBipartite(weight_matrix)


def greedyBipartite(graph):
    (m, n) = graph.shape
    Ml = graph.tolil()
    Mlt = graph.transpose().tolil()
    rowSet = set(range(0, m))
    colSet = set(range(0, n))
    curScore = graph[list(rowSet), :][:, list(colSet)].sum(axis=None)
    bestAveScore = curScore / (len(rowSet) + len(colSet))
    bestSets = (rowSet, colSet)

    # *decrease* in total weight when *removing* this row
    # Prepare the min priority tree to begin greedy algorithm.
    rowDeltas = np.squeeze(graph.sum(axis=1).A)
    colDeltas = np.squeeze(graph.sum(axis=0).A)

    rowTree = MinTree(rowDeltas)
    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    while rowSet and colSet:

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
