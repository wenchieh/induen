import numpy as np
from scipy.sparse import lil_matrix
from src.MaxTree import MaxTree
from src.MinTree import MinTree

""" Give a graph, its aubgraph i.e. an indicator vector, score of the aveage degree
    Expand and contract the subgraph with higher score with the neighbor nodes.
"""

def fastExpander(graph: lil_matrix, sub: list, score):

    # graph: undirected graph adjacency lil_matrix
    # sub: subgraph that to be expand , array_like data
    # score : sum of edge weights / number of nodes

    subgraph = sub.copy()
    curSet = set(subgraph)
    curAveScore = score
    curScore = score * len(subgraph)
    residualNodes = set(range(graph.shape[0])) - curSet
    residualList = sorted(residualNodes)
    residualPrior = np.squeeze(graph[residualList, :][:, subgraph].sum(axis=1).A)
    # Construct the maximum priority tree
    priority = MaxTree(residualPrior)
    isExpand = True

    while isExpand:

        node, prior = priority.getMax()
        if (curScore+prior)/(len(curSet)+1) >= curAveScore:
            curScore += prior
            trueNode = residualList[node]
            curSet.add(trueNode)
            curAveScore = curScore / len(curSet)

            # Change the priority of the node and its neighbors not in curSet.
            priority.changeVal(node, float('-inf'))
            for neigh in graph.rows[trueNode]:
                if neigh not in curSet:
                    delt = graph[trueNode, neigh]
                    tmpNeigh = residualList.index(neigh)
                    priority.changeVal(tmpNeigh, delt)
        else:
            isExpand = False
    subgraph = sorted(curSet)
    return subgraph, curAveScore


def fastContracter(graph: lil_matrix, sub: list, score):

    # graph: undirected graph adjacency lil_matrix
    # sub: subgraph that to be contract , array_like data
    # score : sum of edge weights / number of nodes

    subgraph = sub.copy()
    curSet = set(subgraph)
    curAveScore = score
    curScore = score * len(subgraph)
    curPrior = np.squeeze(graph[subgraph, :][:, subgraph].sum(axis=1).A)
    # Construct the maximum priority tree
    priority = MinTree(curPrior)
    isContract = True

    while isContract:

        node, prior = priority.getMin()
        if (curScore-prior)/(len(curSet)-1) >= curAveScore:
            curScore -= prior
            trueNode = subgraph[node]
            curSet.remove(trueNode)
            curAveScore = curScore / len(curSet)

            # Change the priority of the node and its neighbors in curSet.
            priority.changeVal(node, float('inf'))
            for neigh in graph.rows[trueNode]:
                if neigh in curSet:
                    delt = graph[trueNode, neigh]
                    tmpNeigh = subgraph.index(neigh)
                    priority.changeVal(tmpNeigh, delt)
        else:
            isContract = False
    subgraph = sorted(curSet)
    return subgraph, curAveScore