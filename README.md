# Interrelated Dense Subgraphs Detection in Multilayer Networks

This repo contains the implementations for the **InDuen** Algorithm and detailed information about how to deploy it.

We provide a series of algorithms for detecting interrelated dense subgraphs in multilayered interdependent and multi-partite networks. We show how to deploy these algorithms in detail.

**If the strategy is "layer-by-layer"**, InDuen will detect dense subgraphs in each network separately.

**If the strategy is "joint"**, InDuen will jointly reweight the cross-domain links (i.e., interactions/dependencies in the real-world dataset) and detect dense subgraphs in the multilayered network.

For dense subgraph detector, we will provide **{MaxFlow, Greedy, GreedyOQC, MFDSD, Fraudar, Greedy++, SpecGreedy, CoreApp}** detectors, and Greedy is the default one used in our InDuen algorithm. The MAXFLOWSOLVER mentioned in the paper refers to the MaxFlow method proposed by Goldberg, and the GREEDYSOLVER refers to the Greedy method provided by Charikar. In our experiments, detector$^{a}$ means input the aggregated multilayered network to this detector. Please read our paper to see more details.

**If boost == True**, InDuen will apply the heuristic boosting strategy to improve the performance of the results retrieved by the DSD detector, that is, expand and contract the subgraph based on its neighbors until it converges. By the way, expand and contract operations are heuristic, useful, and efficient for some graph discrete optimization tasks, such as conductance and modularity etc.

**Relationship with SpecGreedy**: If only one layer exists, then this task degenerates to detect the dense subgraph in a single network. If we remove the non-negative constraint of the MF module and ignore the boost module, then the algorithm is almost equivalent to the SpecGreedy algorithm.



Reference
========================
If you use this code as part of any published research, please acknowledge the following papers.
```
@inproceedings{feng2024induen,
  title={Interrelated Dense Subgraphs Detection in Multilayer Networks},
  author={Wenjie Feng, Li Wang, Bryan Hooi, See-Kiong NG, Shenghua Liu},
  booktitle={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
}
