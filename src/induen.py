import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix, csc_matrix

from src.baselines import *
from src.fraudar import Fraudar
from src.conf import coupled_orthogonal_nmf, coupled_nmf
from src.boosting import expander
from src.utils import convert_to_csc,convert_to_lil,select_candidate,sample,aggregate,indicator

""" 
    Parameters in induen
    As : list of within-layer network in networkx
    Cs : dict of cross-layer dependency network in networkx
    GG : the structure of the dependency in multi-layered network
    gammas: edge importance for cross-layer network
    boost : Optional, boosting or not. (Note: we do not recommend boosting for network with huge NNZ.)
    Ortho : with orthogonal constraints or not
    beta  : coupled strength in matrix factorization
    R     : low-rank
    epoch : iteration number
    reg   : l1 regularization
    seed  : random seed

"""

dsd_detectors = {
    "greedy": greedyCharikar,
    "oqc": greedyOqc, 
    "fraudar": Fraudar,
    "mfdsd": mfdsd,
    "coreds": coreds,
    "specgreedy": specgreedy}


class INDUEN():

    def __init__(self, As:list,Cs:dict,GG:np.ndarray,gammas:np.ndarray,strategy="joint",boost=True,beta=None, K=10, epoch=50, reg=1e-10, seed=1234, detector="greedy", factorization='ortho'):
        self.As,self.Cs = convert_to_csc(As,Cs)
        self.GG = GG
        self.gammas = gammas
        self.strategy = strategy
        self.boost = boost
        self.beta = beta
        self.K = K
        self.epoch = epoch
        self.reg = reg
        self.seed = seed
        self.layer = len(As)
        self.factorization = factorization
        self.detector = dsd_detectors[detector]
        self.get_factors()
        self.As,self.Cs = convert_to_lil(As,Cs)

    def get_factors(self):
        if self.factorization == 'ortho':
            self.factors, self.lambdas, self.sigmas, self.beta = coupled_orthogonal_nmf(self.As, self.Cs, self.GG, self.beta, self.K, self.epoch, self.reg, self.seed)
        else:
            self.factors, self.lambdas, self.sigmas, self.beta = coupled_nmf(self.As, self.Cs, self.GG, self.beta, self.K, self.epoch, self.reg, self.seed)

    def change_gammas(self, gammas):
        self.gammas = gammas

    def run(self):
        
        if self.boost:
            self.blockmat,self.totalpos = aggregate(self.As, self.Cs, self.GG, self.gammas)

        bestres = None
        best_score = 0
        allres = []
        nsize = sum([adj.shape[0] for adj in self.As])

        for k in range(self.K):

            candidate = [None] * self.layer
            for i in range(self.layer):
                candidate[i] = select_candidate(self.factors[i], nsize, k)

        
            subAs,subCs = sample(self.As, self.Cs, candidate)
            block, pos = aggregate(subAs,subCs,self.GG,self.gammas)
            density = block.sum() / (2 * block.shape[0]) # average degree density
            print("After candidating, size of each layer: ", [len(cand) for cand in candidate])
            print("The joint density is {:.4f}".format(density))
            res, density = self.detector(block)  
            res = np.array(sorted(res))

            # Split the result into the corresponding nodes in each layer
            finalres = []
            for i in range(self.layer):
                tmp_node_idx = res[(res >= pos[i]) & (res < [pos[i+1]])] - pos[i]
                ori_node_idx = [candidate[i][idx] for idx in list(tmp_node_idx)]
                finalres.append(ori_node_idx)
            print("After dsd detector, size of each layer: ", [len(re) for re in finalres])
            print("The joint density is {:.4f}".format(density))
            
            # Boost for each column vector group.
            if self.boost:
                boostres = []
                for i in range(self.layer):
                    boostres.extend([idx+self.totalpos[i] for idx in finalres[i]])
                boostres, density = expander(self.blockmat, boostres, density)
                boostres = np.array(sorted(list(boostres)))
                # Split the boostres into each layer
                for i in range(self.layer):
                    finalres[i] = list(boostres[(boostres >= self.totalpos[i]) & (boostres < [self.totalpos[i+1]])] - self.totalpos[i])
                
                print("After neighbor boosting, size of each layer: ", [len(re) for re in finalres])
                print("The joint density is {:.4f}".format(density))


            density *= indicator(finalres)

            print("The optimal joint density of the {}-th column of factors: {:.4f}".format(k,density))
            print('\n')

            if density > best_score:
                best_score = density
                bestres = finalres
            allres.append((finalres,density))

        if bestres == None:
            raise ValueError("For each column of factors, there is at least one layer with no nodes return, please set another gammas.")

        return bestres, best_score, allres


    