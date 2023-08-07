import os
import pickle
import numpy as np
import networkx as nx
import scipy.io as scio
from scipy.sparse import lil_matrix, csc_matrix
from src.utils import random_matrix

import sys
sys.path.append("..")

nx2sp = nx.to_scipy_sparse_array
er_model = nx.erdos_renyi_graph 
sf_model = nx.barabasi_albert_graph


"""
Dataset:
    load and preprocess the multilayered network efficiently.
    The dataset includes:
        real-world dataset:
        "Aminer","Bio","Infra3","dblp","REDDA"
        synthetic dataset:
        "ER","SF","subBio"

Params:
    do: the observed cross-layer dependencies
    du: the complete cross-layer dependencies
    g:  the struture of the layer-layer dependencies
    As: list of within-layer adjacency matrix 
    Cs: dict of cross-layer dependency matrix
    As we don't consider the node attributes, we use the scipy.sparse matrix to store the graph.
"""

class Dataset:
    def __init__(self, name, verbose=False, camouflage=False, **kwargs):
        
        self.name = name
        self.threshold = 1e-10 # similarity threshold
        self.camouflage = camouflage

        if self.name in ['Aminer','Bio','Infra3','Infra5']:

            data = scio.loadmat(f'../data/{name}.mat')
            dnew, du, g = data['D_new'], data['DU'], data['G']
            self.layer = len(g[0])
            self.As = [lil_matrix(g[0][i][0][0][0]) for i in range(self.layer)]
            self.size = [adj.shape[0] for adj in self.As]
            self.Cs = dict()
            for idx in range(len(du[0])):
                cs = du[0][idx][0][0][0]
                i = self.size.index(cs.shape[0])
                j = self.size.index(cs.shape[1])
                self.Cs[(i,j)] = lil_matrix(cs)
                self.Cs[(j,i)] = self.Cs[(i,j)].T
            self.GG = dnew.toarray()

            if self.name == 'Bio':
                self.fields = ['Chem','Gene','DZ']
                self.threshold = 0.5

            elif self.name == 'Infra3':
                self.fields = ['AP','AS','Power']

            elif self.name == 'Aminer':
                self.fields = ['Author','Paper','Venue'] # reorder the layers
                tmpAs,tmpCs = self.As.copy(), self.Cs.copy()
                self.As = [tmpAs[2],tmpAs[0],tmpAs[1]]
                self.Cs = {(0,1):tmpCs[(2,0)],(1,0):tmpCs[(0,2)],
                           (1,2):tmpCs[(0,1)],(2,1):tmpCs[(1,0)]}
                self.GG = np.array([[0,1,0],[1,0,1],[0,1,0]])

        elif self.name == 'dblp':

            g1,g2,g3,c12,c23 = pickle.load(open("../casestudy/dblp", "rb"))
            self.As = [g1,g2,g3]
            self.Cs = {(0,1):c12,(1,0):c12.T,
                       (1,2):c23,(2,1):c23.T}
            self.GG = np.array([[0,1,0],[1,0,1],[0,1,0]])
            self.size  = [adj.shape[0] for adj in self.As]
            self.layer = len(self.As)
            self.fields = ['Author','Paper','Venue']
        
        elif self.name in ['ER','SF','subBio']:
            
            self.size  = [1800,2400,3000]
            self.layer = 3
            self.block_num = 10
            self.ms = [40,60,80] # preferential attachment params

            self.prob_bg = 0.05 # background 
            self.prob_related  = 0.05 
            self.prob_unrelated = 0.001 
            self.prob_camouflage = 1.0 # inject clique as camouflage

            self.insert_pos = 0
            self.fake_pos = [180,240,300]
            self.clique_size = [int(size/self.block_num) for size in self.size]
            self.ground_truth = []
            self.As = []
            self.Cs = dict()
            self.fields = ['layer1','layer2','layer3']
            
            
            # generate the within-layer connection and ground truth.
            for i in range(self.layer):
                if self.name=="ER":
                    graph_ = lil_matrix(nx2sp(er_model(n=self.size[i],p=self.prob_bg,seed=1234)))
                elif self.name=="SF":
                    graph_ = lil_matrix(nx2sp(sf_model(n=self.size[i],m=self.ms[i],seed=1234)))
                self.As.append(graph_)
                self.ground_truth.append(frozenset(range(self.insert_pos,self.insert_pos+self.clique_size[i])))
                
            if self.camouflage: 
                # generate the camouflage edges.
                for i, graph_ in enumerate(self.As):
                    graph_[self.clique_size[i]:2*self.clique_size[i],self.clique_size[i]:2*self.clique_size[i]] = lil_matrix(nx2sp(er_model(n=self.clique_size[i],p=self.prob_camouflage,seed=1234)))

            # generate the cross-layer dependency.
            for i in range(self.layer):
                block_i = int(self.As[i].shape[0]/self.block_num)
                for j in range(i+1, self.layer):
                    block_j = int(self.As[j].shape[0]/self.block_num)
                    cs = random_matrix((self.As[i].shape[0], self.As[j].shape[0]), self.prob_unrelated)
                    cs[0:block_i, 0:block_j] = random_matrix((block_i, block_j), self.prob_related)
                    self.Cs[(i,j)] = lil_matrix(cs)
                    self.Cs[(j,i)] = lil_matrix(cs).T
            self.GG = np.ones((3,3), dtype=np.float64)-np.identity(3)

        else:
            raise ValueError("The dataset is not supported now!")

        self.process() # processing
        if verbose:
            self.display()

    @property
    def _data(self):
        return self.As, self.Cs, self.GG, self.fields

    def inject(self, prob_gt=0.1):
        # inject the ground truth.
        for i,graph_ in enumerate(self.As):
            graph_[0:self.clique_size[i], 0:self.clique_size[i]] = lil_matrix(nx2sp(er_model(n=self.clique_size[i], p=prob_gt, seed=1234)))
        return self.As # only change the As.


    # delete the self-loop edges in the As.
    # make the adjacency matrix to be symmetric.
    # make the weight of the edges to be 1.
    def process(self):
        for i in range(self.layer):

            self.As[i][range(self.As[i].shape[0]), range(self.As[i].shape[0])] = 0
            xs,ys = self.As[i].nonzero()

            for x,y in zip(xs,ys):
                self.As[i][x,y] = 1 if self.As[i][x,y] >= self.threshold else 0
            self.As[i][ys,xs] = 1 # make adjacency matrix symmetric.

            for j in range(self.layer):
                if self.GG[i,j] != 0:
                    self.Cs[(i,j)][self.Cs[(i,j)].nonzero()] = 1

    # display the information of the dataset
    def display(self):
        print(f'This is a {self.layer} layered {self.name} network')
        # print the shape and nnz of each network
        for i in range(self.layer):
            print(f"For the {i}-th within layer")
            print("shape: ", self.As[i].shape, "nnz: ", self.As[i].nnz)

        for i,j in self.Cs.keys():
            if j>i:
                print(f"For the {(i,j)}-th cross layer")
                print("shape: ", self.Cs[(i,j)].shape, "nnz: ", self.Cs[(i,j)].nnz)

        total_nnz = 0 
        for adj in self.As:
            total_nnz += adj.nnz
        for _,cs in self.Cs.items():
            total_nnz += cs.nnz 
        print(f"The total nnz of the multilayered network is {total_nnz}")
        print("The layer-layer dependency structure is: ")
        print(self.GG)
