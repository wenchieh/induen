from scipy.sparse import lil_matrix

# Give a graph and its indicator vector,
# Extend the subgraph with higher score with the neighbor nodes.

def neighborBoosting(graph: lil_matrix, sub: list):

    # graph: undirected graph adjacency lil_matrix
    # sub: subgraph that to be extend , array_like data
    # curScore : edge weights / node numbers
    
    subgraph = sub.copy()
    curSet = set(subgraph)
    addSet = set()
    curScore = graph[subgraph, :][:, subgraph].sum() / 2
    curAveScore = curScore / len(curSet)

    # Initialize addSet as all the neighbors of the current subgraph
    for node in subgraph:
        subSet = set(graph.rows[node]) - curSet
        addSet = addSet | subSet

    # Loop until there are no node in addSet can boosting the score
    for neigh in addSet:
        addScore = graph[neigh][:, subgraph].sum()

        if (curScore+addScore) / (len(curSet)+1) >= curAveScore:
            curScore += addScore
            curSet.add(neigh)
            curAveScore = curScore / len(curSet)

            subgraph.append(neigh)
            subSet = set(graph.rows[neigh]) - curSet
            addSet = addSet | subSet

    return subgraph, curAveScore
