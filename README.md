# Interrelated Dense Subgraphs Detection in Multilayered Networks

This repo contains the implementations for the **IntDuen** Algorithm and detailed information about how to deploy it.

We provide a series of algorithms for detecting interrelated dense subgraphs in multilayered interdependent network and multi-partite network. We show how to deploy these algorithms in detail.

**If the strategy is "layer-by-layer"**, IntDuen will detect dense subgraph in each network separately.

**If the strategy is "joint"**, IntDuen will reweight the cross-domain links (i.e. interactions/dependencies in real-world dataset) and detect dense subgraph in the multilayered network jointly.

For dense subgraph detector, we will provide **{MaxFlow, Greedy, GreedyOQC, MFDSD, Fraudar, Greedy++}** detectors, and Greedy is the default one used in our IntDuen algorithm. The MAXFLOWSOLVER mentioned in paper refers to the MaxFlow method proposed by Goldberg, and the GREEDYSOLVER refers to the Greedy method provided by Charikar. In our experiments, detector$^{a}$ means input the aggregated multilayered network to this detector. Please read our paper to see more details.

**If boost == True**, IntDuen will apply the heuristic boosting strategy to improve the performance of the results retrived by the DSD detecor, that is, expand and contract the subgraph based on its neighbors until converge. By the way, expand and contract operations are heuristic, useful and efficient for some graph discrete optimization task, such as conductance and modularity, etc.

**Relationship with SpecGreedy**: If there are only one layer, then this task degenerates to detect the dense subgrpah in a single network. If we remove the non-negetive constraint of the MF module and ignore the boost module, then the algorithm is almost equivalent to the SpecGreedy algorithm.
